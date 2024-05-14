# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/12/5
import time
from typing import Optional

import torch
import wandb
from datasets import load_dataset, enable_caching
from fire import Fire
from numpy import asarray
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    get_linear_schedule_with_warmup
)

from models import MlpForImageClassification
from newton_optim import (
    add_hooks_,
    disable_hooks_,
    enable_hooks_,
    clear_backprops_,
    rescale_gradient_for_all_,
)

set_seed(233)

enable_caching()


def _collate_fn(batch):
    return {k: torch.tensor([example[k] for example in batch]) for k in batch[0].keys()}


def train_mlp_on_mnist(
        enable_newton: Optional[bool] = True,
        use_adam: Optional[bool] = False,
        profile_hessian_per_steps: Optional[int] = 0,
        delay_newton_steps: Optional[int] = 0,
        # training arguments
        batch_size: Optional[int] = 4096,
        num_epochs: Optional[int] = 2,
        learning_rate: Optional[float] = 0.01,
        weight_decay: Optional[float] = 0.,
        warmup_steps: Optional[int] = 10,
        evaluation_steps: Optional[int] = 100,
        normalization_epsilon: Optional[float] = 5e-3,
        quantum_sparsity_ratio: Optional[float] = 1.0,
        # model arguments
        hidden_size: Optional[int] = 16,
        num_hidden_layers: Optional[int] = 2,
):
    if not enable_newton:
        delay_newton_steps = -1
    mnist_dataset = load_dataset("mnist")
    train_dataset = mnist_dataset["train"]
    eval_dataset = mnist_dataset["test"]

    train_dataset = train_dataset.map(
        lambda x: {"pixel_values": asarray(x["image"], dtype=float).reshape(1, 28, 28).tolist(), "labels": x["label"]},
        remove_columns=["image", "label"],
    )
    eval_dataset = eval_dataset.map(
        lambda x: {"pixel_values": asarray(x["image"], dtype=float).reshape(1, 28, 28).tolist(), "labels": x["label"]},
        remove_columns=["image", "label"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=8
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=8
    )

    model = MlpForImageClassification(
        num_classes=10, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers
    )

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if use_adam else SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs * len(train_dataloader)
    )

    wandb.init(
        project="qnewton",
        name=f"train-mlp-{hidden_size}-mnist" + ("-with-newton" if enable_newton else ""),
    )

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total training steps: {len(train_dataloader) * num_epochs}")
    print(f"Total epochs: {num_epochs}")
    print(f"Warmup steps: {warmup_steps}")

    # train!
    best_eval_loss = float('inf')
    if enable_newton:
        has_hooks = True
        add_hooks_(model)
        print('Enable Newton')
    else:
        has_hooks = False
    model.train()
    model = model.cuda()
    total_step = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), desc='Training')
    timeit_total = {'forward-backward': 0, 'hessian': 0, 'inversion': 0, 'rescale': 0}
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            enable_newton = (total_step >= delay_newton_steps) and (delay_newton_steps >= 0)
            batch = {k: v.cuda() for k, v in batch.items()}
            if has_hooks:
                clear_backprops_(model)
            start = time.time()
            torch.cuda.synchronize()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.cuda.synchronize()
            end = time.time()
            # progress_bar.write(f'[timeit] forward + gradient backward: {round(end - start, 3)} s')
            timeit_total['forward-backward'] += end - start

            # newton
            if enable_newton:
                step_timeit_dict = rescale_gradient_for_all_(
                    model,
                    damping=1e-10,
                    max_number_of_params=200000000,  # 2e4
                    disable_warning=(total_step != 0),
                    print_fn=lambda x: progress_bar.write(f"[step {total_step}]" + x),
                    profile_hessian=(profile_hessian_per_steps > 0 and total_step % profile_hessian_per_steps == 0),
                    normalization_epsilon=normalization_epsilon,
                    quantum_sparsity_ratio=quantum_sparsity_ratio
                )
                print(step_timeit_dict)
                for k, v in step_timeit_dict.items():
                    timeit_total[k] += v

            optimizer.step()
            optimizer.zero_grad()
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
            }, step=total_step)
            total_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {'loss': loss.item(),
                 'learning_rate': scheduler.get_last_lr()[0],
                 }.update({k: round(v, 3) for k, v in timeit_total.items()}))

            # wandb.log({
            #     "time-" + k: v / total_step for k, v in timeit_total.items()
            # }, step=total_step)

            if (total_step + 1) % evaluation_steps == 0:
                model.eval()
                disable_hooks_()
                eval_loss = 0
                all_preds = []
                all_labels = []
                for eval_batch in eval_dataloader:
                    eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
                    with torch.no_grad():
                        eval_outputs = model(**eval_batch)

                    all_preds.append(eval_outputs.logits.argmax(dim=-1))
                    all_labels.append(eval_batch['labels'])
                    eval_loss += eval_outputs.loss.item()
                eval_loss /= len(eval_dataloader)
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                eval_accuracy = (all_preds == all_labels).float().mean().item()

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                wandb.log({
                    'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                })
                wandb.summary['best_eval_loss'] = best_eval_loss
                model.train()
                enable_hooks_()

        scheduler.step()


if __name__ == '__main__':
    Fire(train_mlp_on_mnist)
