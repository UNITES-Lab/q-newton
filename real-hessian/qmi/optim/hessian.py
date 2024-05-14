# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/10/21
import time
from typing import Dict, List

import jax
import jax.numpy as jnp
import wandb
from flax.traverse_util import flatten_dict, unflatten_dict
from tqdm import tqdm


@jax.jit
def _numel(x):
    return jnp.prod(jnp.array(x.shape))


def compute_vector_cum_percent_numel(vector: jnp.ndarray, percentages: List[float]) -> List[jnp.ndarray]:
    vector = jnp.abs(vector)
    vector = vector / jnp.sum(vector)
    vector = jnp.sort(vector)[::-1]
    cum_numels = []
    for percentage in percentages:
        cum_vector = jnp.cumsum(vector)
        num_elements = jnp.argmax(cum_vector >= percentage)
        cum_numels.append(num_elements.item() + 1)
    return cum_numels


def prune_matrix_for_row_sparsity(
        matrix: jnp.ndarray,
        row_sparsity: int,
):
    """
    Examples
    --------
    >>> from jax import numpy as jnp
    >>> matrix = jnp.array([[1, 2, -3], [4, -5, 6], [7, 8, -9]], dtype=jnp.float32)
    >>> prune_matrix_for_row_sparsity(matrix, 2)
    Array([[ 0.,  2., -3.],
           [ 0., -5.,  6.],
           [ 0.,  8., -9.]], dtype=float32)
    """
    # select top-row_sparsity elements in each row and set the rest to 0
    # matrix: (size, size)
    size = matrix.shape[0]
    assert row_sparsity <= size
    # (size, size)
    sorted_indices = jnp.argsort(jnp.abs(matrix), axis=1)
    # (size, row_sparsity)
    top_indices = sorted_indices[:, -row_sparsity:]
    # (size, size)
    mask = jnp.zeros_like(matrix)
    mask = mask.at[jnp.arange(size)[:, None], top_indices].set(1)
    return matrix * mask


def compute_newton_gradient(
        grads: Dict,
        hessian: Dict,
        hessian_row_sparsity_ratio: float = 1.0,
) -> Dict:
    """
    Compute the Newton gradient using the Hessian matrix and the gradient vector.

    Parameters
    ----------
    grads : Dict
        The gradient vector.
    hessian : Dict
        The Hessian matrix.
    hessian_row_sparsity_ratio : float
        The ratio of the number of non-zero elements in each row of the Hessian matrix to the number of elements in each
        row of the Hessian matrix. The larger the ratio, the more sparse the Hessian matrix is.

    Returns
    -------
    newton_gradient_dict : Dict
        The Newton gradient.
    """
    # Nested dictionary flattening
    total_start_time = time.time()
    grads = flatten_dict(grads)
    grads_flat = jax.tree_map(lambda x: x.reshape(-1), grads)  # flatten gradients into a vector
    hessian_flat = flatten_dict(hessian)

    gradient_concat = jnp.concatenate([grads_flat[key] for key in grads_flat.keys()])
    param_keys = grads_flat.keys()
    hessian_concat = jnp.concatenate([
        jnp.concatenate([
            hessian_flat[key1 + key2].reshape(grads_flat[key1].shape[0], grads_flat[key2].shape[0]) for key2 in
            param_keys
        ], axis=1) for key1 in param_keys
    ])

    # regularization
    max_val = jnp.max(jnp.abs(hessian_concat))
    hessian_concat = hessian_concat + max_val * 5e-3 * jnp.eye(hessian_concat.shape[0])
    hessian_concat = prune_matrix_for_row_sparsity(
        hessian_concat, int(hessian_concat.shape[0] * hessian_row_sparsity_ratio)
    )

    # tqdm.write(f"[Debug]Condition number of hessian matrix: {jnp.linalg.cond(hessian_concat)}")

    hessian_concat = hessian_concat.block_until_ready()
    start_time = time.time()
    hessian_inv = jnp.linalg.inv(hessian_concat).block_until_ready()
    end_time = time.time()
    hessian_inversion_time = end_time - start_time

    newton_gradient = hessian_inv @ gradient_concat

    # map newton gradient back to the original structure
    newton_gradient_dict = {}
    start_idx = 0
    for key in param_keys:
        end_idx = start_idx + grads_flat[key].shape[0]
        newton_gradient_dict[key] = newton_gradient[start_idx:end_idx].reshape(grads[key].shape)
        start_idx = end_idx
    newton_gradient_dict = unflatten_dict(newton_gradient_dict)

    newton_gradient_dict = jax.tree_map(lambda x: x.block_until_ready(), newton_gradient_dict)
    total_end_time = time.time()
    tqdm.write(
        f"[Profile] Hessian inversion time: {round(hessian_inversion_time, 4)} seconds | "
        f"Gradient rescaling time: {round(total_end_time - total_start_time - hessian_inversion_time, 4)} seconds")

    return newton_gradient_dict


def compute_newton_gradient_layer_wise(
        grads: Dict,
        hessian: Dict,
        hessian_row_sparsity_ratio: float = 1.0,
) -> Dict:
    """
    Only compute the Newton gradient for each layer. If the condition number of the Hessian matrix is larger than
    `condition_number_threshold`, then the Newton gradient is not computed for this layer.

    i.e. For each tree leaf (name_keys) in `grads`, rescale the gradient by the inverse of the only tree leaf
        (name_keys, name_keys) in `hessian`.
    """
    total_start_time = time.time()
    grads = flatten_dict(grads)
    grads_flat = jax.tree_map(lambda x: x.reshape(-1), grads)  # flatten gradients into a vector
    hessian_flat = flatten_dict(hessian)
    newton_rescaled_gradient_dict = {}
    hessian_inversion_time = 0
    for key in grads_flat:
        num_params = grads_flat[key].shape[0]
        layer_hessian = hessian_flat[key + key].reshape(num_params, num_params)

        max_val = jnp.max(jnp.abs(layer_hessian)) + 1e-6  # add a small value to avoid numerical issues
        layer_hessian = layer_hessian + max_val * 5e-3 * jnp.eye(num_params)
        layer_hessian = prune_matrix_for_row_sparsity(layer_hessian, int(num_params * hessian_row_sparsity_ratio))

        # hessian_cond = jnp.linalg.cond(layer_hessian)
        if key == ('Dense_0', 'kernel'):
            singulars = jnp.linalg.svd(layer_hessian, compute_uv=False)
            singular_log = {
                "max_singular": singulars[0].item(),
                "min_singular": singulars[-1].item(),
                "condition_number": (singulars[0] / singulars[-1]).item()
            }
            wandb.log(singular_log)
            tqdm.write(f"[Profile] Singular values: max={singulars[0].item()} min={singulars[-1].item()} ")

        layer_hessian = layer_hessian.block_until_ready()
        start_time = time.time()
        hessian_inv = jnp.linalg.inv(layer_hessian).block_until_ready()
        end_time = time.time()
        hessian_inversion_time += end_time - start_time
        # percentiles = compute_vector_cum_percent_numel(layer_hessian.flatten(), [0.25, 0.5, 0.75])
        # percentiles = [jnp.sqrt(x).item() for x in percentiles]

        # tqdm.write(
        #     f"[Profile] Hessian of size={num_params} & condition number={round(hessian_cond, 1)} & sparsity={percentiles}"
        #     f"takes time {round(end_time - start_time, 4)}s"
        # )
        # if num_params > 10000:
        #     print(f"{round(hessian_cond, 1)} {percentiles} {round(end_time - start_time, 4)}")

        newton_rescaled_gradient = hessian_inv @ grads_flat[key]
        newton_rescaled_gradient_dict[key] = newton_rescaled_gradient.reshape(grads[key].shape)
    newton_rescaled_gradient_dict = unflatten_dict(newton_rescaled_gradient_dict)
    newton_rescaled_gradient_dict = jax.tree_map(lambda x: x.block_until_ready(), newton_rescaled_gradient_dict)
    total_end_time = time.time()
    # tqdm.write(f"[Profile] Hessian inversion time: {round(hessian_inversion_time, 4)} s | "
    #            f"Gradient rescaling time: {round(total_end_time - total_start_time - hessian_inversion_time, 4)} s")
    return newton_rescaled_gradient_dict
