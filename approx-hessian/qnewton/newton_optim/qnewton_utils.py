# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/1/6
from math import e, pi, log, sqrt

import torch


def prune_matrix_for_quantum_sparsity(
        matrix: torch.Tensor,
        quantum_sparsity: int,
):
    """
    Parameters
    ----------
    matrix: (size, size)
        The matrix to be pruned.
    quantum_sparsity: int
        The number of non-zero elements in each row.

    Returns
    -------
    matrix: (size, size)
        The pruned matrix.

    Examples
    --------
    >>> matrix = torch.tensor([[1, 2, -3], [4, -5, 6], [7, 8, -9]], dtype=torch.float32)
    >>> prune_matrix_for_quantum_sparsity(matrix, 2)
    tensor([[ 0.,  2., -3.],
            [ 0., -5.,  6.],
            [ 0.,  8., -9.]])
    """
    # select top-k elements in each row and set the rest to 0
    # matrix: (size, size)
    size = matrix.shape[0]
    assert quantum_sparsity <= size
    # (size, size)
    sorted_indices = torch.argsort(matrix.abs(), dim=1)
    # (size, row_sparsity)
    top_indices = sorted_indices[:, -quantum_sparsity:]
    # (size, size)
    mask = torch.zeros_like(matrix)
    mask = mask.scatter(dim=1, index=top_indices, src=torch.ones_like(matrix))
    return matrix * mask


def compute_expected_query_cost(
        alpha: float,
        kappa: float,
        epsilon: float,
) -> float:
    query_cost = (1741 * alpha * e / 500) * sqrt(kappa ** 2 + 1) * (
            (133 / 125 + 4 / (25 * kappa ** (1 / 3))) * pi * log(2 * kappa + 3) + 1) + (351 / 50) * log(
        2 * kappa + 3) ** 2 * (log(451 * log(2 * kappa + 3) ** 2 / epsilon) + 1) + alpha * kappa * log(32 / epsilon)
    return alpha * query_cost / (0.39 - 0.204 * epsilon)
