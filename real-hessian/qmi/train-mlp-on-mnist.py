# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/8/28
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable

import jax
import optax
import wandb
from datasets.arrow_dataset import Dataset
from fire import Fire
from flax.training import train_state
from jax import numpy as jnp
from tqdm import tqdm

from data import get_mnist_datasets
from models import MlpForImageClassification
from optim import compute_newton_gradient, compute_newton_gradient_layer_wise


def collate_fn(examples):
    pixel_values = jnp.array([example["pixel_values"] for example in examples])
    labels = jnp.array([example["labels"] for example in examples])
    batch = {"pixel_values": pixel_values, "labels": labels}
    return batch


def create_learning_rate_fn(
        train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


@dataclass
class TrainingConfig:
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 1e-3
    num_epochs: Optional[int] = 10
    seed: Optional[int] = 42


def init_train_state(
        rng: jnp.ndarray,
        model: MlpForImageClassification,
        learning_rate_fn,
        shape: Tuple[int, ...],
        adam: bool = True,
) -> train_state.TrainState:
    variables = model.init(rng, jnp.ones(shape, jnp.float32))
    if adam:
        tx = optax.adam(learning_rate_fn)
    else:
        tx = optax.sgd(learning_rate_fn)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


@jax.jit
def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def get_train_data_collator(
        rng: jnp.ndarray,
        dataset: Dataset,
        batch_size: int,
):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        yield batch


@jax.jit
def sgd_train_step(
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    targets = batch.pop("labels")
    images = batch.pop("pixel_values")

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, images
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, 10))
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics


def sgd_train_step_with_newton(
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        layer_wise: bool,
        hessian_row_sparsity_ratio: float = 1.0,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    targets = batch.pop("labels")
    images = batch.pop("pixel_values")

    @jax.jit
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, images
        )
        loss = jnp.mean(
            optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, 10))
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    hessian_fn = jax.hessian(loss_fn, has_aux=True)

    grad_start_time = time.time()
    (loss, _), grads = grad_fn(state.params)
    grads = jax.tree_map(lambda x: x.block_until_ready(), grads)
    grad_end_time = time.time()

    hessian_start_time = time.time()
    hessian, _ = hessian_fn(state.params)
    hessian = jax.tree_map(lambda x: x.block_until_ready(), hessian)
    hessian_end_time = time.time()
    # tqdm.write(f"[Profile] Gradient calculation time: {grad_end_time - grad_start_time} seconds"
    #            f" | Hessian calculation time: {hessian_end_time - hessian_start_time} seconds")

    # newton gradient
    if layer_wise:
        grads = compute_newton_gradient_layer_wise(
            grads, hessian,
            hessian_row_sparsity_ratio=hessian_row_sparsity_ratio
        )
    else:
        grads = compute_newton_gradient(
            grads, hessian,
            hessian_row_sparsity_ratio=hessian_row_sparsity_ratio
        )
    # gradient clipping
    grads = jax.tree_map(lambda x: jnp.clip(x, -100, 100), grads)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics


def train_on_mnist_with_sgd(
        hidden_size: Optional[int] = 16,
        num_hidden_layers: Optional[int] = 1,
        batch_size: Optional[int] = 4096,
        learning_rate: Optional[float] = 0.01,
        num_epochs: Optional[int] = 2,
        warmup_steps: Optional[int] = 10,
        using_hessian: Optional[bool] = True,
        layer_wise: Optional[bool] = True,
        hessian_row_sparsity_ratio: Optional[float] = 1.0,
        disable_wandb: Optional[bool] = False,
):
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    optim_name = "sgd"
    run_name = f"mlp-{hidden_size}-mnist-{optim_name}-lr-{learning_rate}-bsz-{batch_size}"
    if using_hessian:
        run_name = run_name + "-newton"
        run_name = run_name + "-layer-wise" if layer_wise else run_name + "-full"
        run_name = run_name + f"-{hessian_row_sparsity_ratio}"

    if not disable_wandb:
        wandb.init(
            project="qnewton",
            config=training_config.__dict__,
            name=run_name,
        )
    train_dataset, test_dataset = get_mnist_datasets()
    model = MlpForImageClassification(num_classes=10, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    rng = jax.random.PRNGKey(training_config.seed)

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        batch_size,
        num_epochs,
        warmup_steps,
        learning_rate,
    )

    state = init_train_state(rng, model, linear_decay_lr_schedule_fn, (training_config.batch_size, 1, 28, 28),
                             adam=False)

    print(f"***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {training_config.num_epochs}")
    print(f"  Train batch size = {training_config.batch_size}")
    print(f"  Total model parameters = {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}")
    if using_hessian:
        print(f"  Using Newton gradient w/ hessian row sparsity ratio = {hessian_row_sparsity_ratio}")

    # train_data_loader = DataLoader(
    #     train_dataset,
    #     batch_size=training_config.batch_size,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    #     num_workers=0,
    #     # persistent_workers=True,
    #     drop_last=True,
    # )

    # Train!
    for epoch in range(training_config.num_epochs):
        rng, input_rng = jax.random.split(rng)
        train_data_collator = get_train_data_collator(input_rng, train_dataset, training_config.batch_size)
        for step, batch in enumerate(
                tqdm(train_data_collator, total=len(train_dataset) // training_config.batch_size)
        ):
            if using_hessian:
                state, train_metrics = sgd_train_step_with_newton(
                    state, batch, layer_wise,
                    hessian_row_sparsity_ratio=hessian_row_sparsity_ratio
                )
            else:
                state, train_metrics = sgd_train_step(state, batch)
            if not disable_wandb:
                wandb.log(train_metrics, step=step + epoch * len(train_dataset) // training_config.batch_size)
            tqdm.write(
                f"Epoch {epoch} | Step {step} | Learning rate: {linear_decay_lr_schedule_fn(state.step)} | Train Loss: {train_metrics['loss']:.3f}"
            )
            if train_metrics['loss'] > 200:
                tqdm.write("Loss is too large, stop training!")
                break

    # Evaluation
    rng, input_rng = jax.random.split(rng)
    test_data_collator = get_train_data_collator(input_rng, test_dataset, training_config.batch_size)
    accuracy_list = []
    for batch in tqdm(test_data_collator, total=len(test_dataset) // training_config.batch_size):
        targets = batch.pop("labels")
        images = batch.pop("pixel_values")
        logits = state.apply_fn(
            {"params": state.params}, images
        )
        accuracy = compute_accuracy(logits, targets)
        accuracy_list.append(accuracy)
    accuracy = jnp.mean(jnp.stack(accuracy_list))
    tqdm.write(f"Test accuracy: {accuracy:.3f}")
    if not disable_wandb:
        wandb.log({"test_accuracy": accuracy})


if __name__ == "__main__":
    Fire(train_on_mnist_with_sgd)
