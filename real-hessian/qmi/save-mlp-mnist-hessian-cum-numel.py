# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/10/21
from dataclasses import dataclass
from typing import Dict
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
import optax
from datasets import Dataset
from fire import Fire
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
from tqdm import tqdm

from data import get_mnist_datasets
from qmi.models import MlpForImageClassification


@dataclass
class TrainingConfig:
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 1e-3
    num_epochs: Optional[int] = 10
    seed: Optional[int] = 42


def init_train_state(
        rng: jnp.ndarray,
        model: MlpForImageClassification,
        learning_rate: float,
        shape: Tuple[int, ...],
        adam: bool = True,
) -> train_state.TrainState:
    variables = model.init(rng, jnp.ones(shape, jnp.float32))
    if adam:
        tx = optax.adam(learning_rate)
    else:
        tx = optax.sgd(learning_rate)
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


def compute_vector_cum_percent_numel(vector: jnp.ndarray, percentages: List[float]) -> List[jnp.ndarray]:
    """
    Compute the number of contribution elements for each cumulative percentage.

    Examples
    --------
    >>> compute_vector_cum_percent_numel(jnp.array([0.1, 0.2, 0.3, 0.4]), [0.5, 0.9])
    [2, 3]
    """
    vector = jnp.abs(vector)
    vector = vector / jnp.sum(vector)
    vector = jnp.sort(vector)[::-1]
    cum_numels = []
    for percentage in percentages:
        cum_vector = jnp.cumsum(vector)
        num_elements = jnp.argmax(cum_vector >= percentage)
        cum_numels.append(num_elements.item() + 1)
    return cum_numels


@jax.jit
def _numel(x):
    return jnp.prod(jnp.array(x.shape))


def compute_newton_gradient_layer_wise_and_get_hessian_row_cum_percent_numel(
        grads: Dict,
        hessian: Dict,
        percentages: List[float]
) -> Tuple:
    # Nested dictionary flattening
    grads_flat = flatten_dict(jax.tree_map(lambda x: x.reshape(-1), grads))  # flatten gradients into a vector
    hessian_flat = flatten_dict(hessian)
    for key in hessian_flat.keys():
        # reshape Hessian to 2D matrix
        hessian_flat[key] = hessian_flat[key].reshape(len(grads_flat[key[:2]]), len(grads_flat[key[2:]]))

    # layer-wisely compute the hessian and newton gradient
    newton_gradient_dict = {}
    grads_flat_unshaped = flatten_dict(grads)
    hessian_row_cum_percent_numel = {
        q: [] for q in percentages
    }
    for layer_key in grads.keys():
        hessian_matrix = []
        keys = list(filter(lambda x: x[0] == layer_key and x[2] == layer_key, list(hessian_flat.keys())))
        layer_num_groups = len(grads[layer_key])
        for row in range(layer_num_groups):
            hessian_row = jnp.concatenate(
                [hessian_flat[keys[layer_num_groups * row + col]] for col in range(layer_num_groups)], axis=1)
            hessian_matrix.append(hessian_row)
        hessian_matrix = jnp.concatenate(hessian_matrix, axis=0)
        gradient_vector = jnp.concatenate([grads_flat[key] for key in grads_flat.keys() if key[0] == layer_key])

        # save hessian matrix and move to cpu
        for idx in range(hessian_matrix.shape[0]):
            cum_numels = compute_vector_cum_percent_numel(hessian_matrix[idx], percentages)
            for q, cum_numel in zip(percentages, cum_numels):
                hessian_row_cum_percent_numel[q].append(cum_numel)

        # regularization
        max_val = jnp.max(jnp.abs(hessian_matrix))
        hessian_matrix = hessian_matrix + max_val * 5e-3 * jnp.eye(hessian_matrix.shape[0])

        # tqdm.write(f"[Debug]Condition number of {layer_key} hessian matrix: {jnp.linalg.cond(hessian_matrix)}")
        hessian_inverse = jnp.linalg.inv(hessian_matrix)
        newton_gradient = hessian_inverse @ gradient_vector

        # unconcactenate the newton gradient into a dictionary and reshape
        for key in filter(lambda x: x[0] == layer_key, grads_flat_unshaped.keys()):
            numel_param = _numel(grads_flat_unshaped[key])
            newton_gradient_dict[key] = newton_gradient[:numel_param].reshape(grads_flat_unshaped[key].shape)
            newton_gradient = newton_gradient[numel_param:]

    newton_gradient_dict = unflatten_dict(newton_gradient_dict)
    return newton_gradient_dict, hessian_row_cum_percent_numel


def sgd_train_step_with_layer_wise_newton(
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    targets = batch.pop("label")
    images = batch.pop("image")

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
    (loss, _), grads = grad_fn(state.params)
    hessian, _ = hessian_fn(state.params)
    # newton gradient
    grads, cum_numel = compute_newton_gradient_layer_wise_and_get_hessian_row_cum_percent_numel(
        grads, hessian, [0.5, 0.75, 0.9, 0.95]
    )
    jnp.save(f"visualizations/cum-numel-step-{state.step}.npy", cum_numel)
    # gradient clipping
    grads = jax.tree_map(lambda x: jnp.clip(x, -100, 100), grads)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics


def train_mlp_on_mnist_and_save_hessian(
        batch_size: Optional[int] = 4096,
        learning_rate: Optional[float] = 0.01,
        num_epochs: Optional[int] = 2,
):
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    train_dataset, test_dataset = get_mnist_datasets()
    model = MlpForImageClassification(num_classes=10)
    rng = jax.random.PRNGKey(training_config.seed)
    state = init_train_state(rng, model, training_config.learning_rate, (training_config.batch_size, 1, 28, 28),
                             adam=False)

    # Train!
    for epoch in range(training_config.num_epochs):
        rng, input_rng = jax.random.split(rng)
        train_data_collator = get_train_data_collator(input_rng, train_dataset, training_config.batch_size)
        for step, batch in enumerate(
                tqdm(train_data_collator, total=len(train_dataset) // training_config.batch_size)
        ):
            state, train_metrics = sgd_train_step_with_layer_wise_newton(state, batch)
            tqdm.write(f"Epoch {epoch} | Step {step} | Train Loss: {train_metrics['loss']:.3f}")
            if train_metrics['loss'] > 200:
                tqdm.write("Loss is too large, stop training!")
                break


if __name__ == "__main__":
    Fire(train_mlp_on_mnist_and_save_hessian)
