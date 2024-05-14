# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/7/10
"""
Simple 2-layer MLP model for MNIST classification
"""

from functools import partial

from flax import linen as nn
from jax import numpy as jnp


class MlpForImageClassification(nn.Module):
    num_classes: int
    hidden_size: int = 16
    num_hidden_layers: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


mlp_toy_for_mnist = partial(MlpForImageClassification, num_classes=10)
