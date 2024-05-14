# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/3
import jax
from datasets import Dataset
from jax import numpy as jnp


def get_train_data_loader(
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
