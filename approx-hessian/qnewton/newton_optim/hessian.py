# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/26
import time
from math import floor
from typing import Dict, Callable, Optional, List

import torch
import wandb
from torch.nn import Module

from .autograd_utils import _layer_type, compute_grad_ps_, _supported_layers
from .qnewton_utils import prune_matrix_for_quantum_sparsity

__all__ = [
    'compute_empirical_fisher_',
    'inverse_fisher_',
    'rescale_gradient_by_fisher_inv_',
]


def _compute_bacthed_vector_cum_percent_numel(
        bacthed_vector: torch.Tensor,
        percentile: float,
) -> List[int]:
    """
    Examples
    --------
    >>> _compute_bacthed_vector_cum_percent_numel(torch.tensor([[0.1, 0.1, 0.1, 0.3], [0.2, 0.2, 0.2, 0.2]]), 0.5)
    [1, 2]
    """
    if bacthed_vector.ndim == 1:
        bacthed_vector = bacthed_vector.unsqueeze(0)
    assert bacthed_vector.ndim == 2, "bacthed_vector must be 2-dim"
    assert 0. <= percentile <= 1., "percentile must be in [0, 1]"

    num_elements = []
    bacthed_vector = torch.abs(bacthed_vector)
    bacthed_vector = bacthed_vector / bacthed_vector.sum(dim=1, keepdim=True)
    bacthed_vector = torch.sort(bacthed_vector, descending=True, dim=1)[0]
    cum_vector = bacthed_vector.cumsum(dim=1)
    for row in range(bacthed_vector.shape[0]):
        num_elements.append(torch.argmax((cum_vector[row] >= percentile).int()).item() + 1)
    return num_elements


def compute_condition_number_upper_bound(
        matrix: torch.Tensor,
) -> float:
    """
    Examples
    --------
    >>> compute_condition_number_upper_bound(torch.tensor([[1., 2.], [3., 4.]]))
    tensor(2.5000)
    """
    assert matrix.ndim == 2, "matrix must be 2-dim"
    assert matrix.shape[0] == matrix.shape[1], "matrix must be square"

    n = matrix.shape[0]
    term = torch.sqrt(
        1 - (
                (1 - n / torch.linalg.norm(matrix, ord='fro')) ** n * (torch.linalg.det(matrix) ** 2)
        )
    )
    return torch.sqrt((1 + term) / (1 - term)).item()


def compute_empirical_fisher_(
        model: Module,
):
    """
    Using empirical Fisher information matrix to rescale gradient.

    Remove attribute 'grad_ps' and add attribute 'fisher' to each layer.

    """
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue

        grad_per_sample = layer.weight.grad_ps
        assert grad_per_sample is not None, "grad_ps not found, call compute_grad_ps(model) first"
        batch_size = grad_per_sample.shape[0]

        grad_per_sample = grad_per_sample.reshape(batch_size, -1)
        # outer product + average over batch
        fisher = torch.einsum('ni,nj->ij', grad_per_sample, grad_per_sample) / batch_size
        setattr(layer.weight, 'fisher', fisher)
        delattr(layer.weight, 'grad_ps')

        if layer.bias is not None:
            grad_per_sample = layer.bias.grad_ps.reshape(batch_size, -1)
            fisher = torch.einsum('ni,nj->ij', grad_per_sample, grad_per_sample) / batch_size
            setattr(layer.bias, 'fisher', fisher)
            delattr(layer.bias, 'grad_ps')


def inverse_fisher_(
        model: Module,
        damping: float = 1e-3,
):
    """
    Compute inverse Fisher information matrix.

    Remove attribute 'fisher' and add attribute 'inv_fisher' to each layer.

    """
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue

        fisher = layer.weight.fisher
        assert fisher is not None, "Fisher not found, call compute_empirical_fisher(model) first"
        inv_fisher = torch.inverse(fisher + torch.eye(
            fisher.shape[0], device=fisher.device, dtype=fisher.dtype) * damping)
        setattr(layer.weight, 'inv_fisher', inv_fisher)
        delattr(layer.weight, 'fisher')

        if layer.bias is not None:
            fisher = layer.bias.fisher
            inv_fisher = torch.inverse(fisher + torch.eye(
                fisher.shape[0], device=fisher.device, dtype=fisher.dtype) * damping)
            setattr(layer.bias, 'inv_fisher', inv_fisher)
            delattr(layer.bias, 'fisher')


def rescale_gradient_by_fisher_inv_(
        model: Module
):
    """
    Rescale gradient by inverse Fisher information matrix.

    """
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue

        grad = layer.weight.grad
        param_shape = grad.shape
        assert grad is not None, "grad not found, call loss.backward() first"
        inv_fisher = layer.weight.inv_fisher
        assert inv_fisher is not None, "inv_fisher not found, call inverse_fisher(model) first"
        grad = grad.flatten()
        grad = torch.einsum('ij,j->i', inv_fisher, grad)
        layer.weight.grad = grad.reshape(param_shape)
        delattr(layer.weight, 'inv_fisher')

        if layer.bias is not None:
            grad = layer.bias.grad
            inv_fisher = layer.bias.inv_fisher
            grad = torch.einsum('ij,j->i', inv_fisher, grad)
            layer.bias.grad = grad
            delattr(layer.bias, 'inv_fisher')


def rescale_gradient_for_all_(
        model: Module,
        damping: Optional[float] = 1e-6,
        max_number_of_params: Optional[int] = 1e8,
        disable_warning: Optional[bool] = True,
        print_fn: Optional[Callable] = print,
        profile_hessian: Optional[bool] = False,
        quantum_sparsity_ratio: Optional[float] = 1.0,
        normalization_epsilon: Optional[float] = 5e-3,
) -> Dict[str, float]:
    """Unified interface for rescaling gradient by Fisher information matrix. Will return timeit in dict."""
    timeit_dict = {'hessian': 0., 'inversion': 0., 'rescale': 0.}

    start = time.time()
    compute_grad_ps_(model, remove_activations=True, remove_backprops=True)
    end = time.time()
    timeit_dict['hessian'] += end - start

    ffn_layer_hessian_condition_numbers_dict = {}
    ffn_layer_hessian_half_percent_numel_dict = {}

    all_condition_numbers = [] if profile_hessian else None
    all_num_params = [] if profile_hessian else None
    for name, layer in model.named_modules():
        layer_type = _layer_type(layer)
        num_params = sum([p.numel() for p in layer.parameters()])
        if layer_type not in _supported_layers or num_params > max_number_of_params:
            # skip large layers
            if num_params > max_number_of_params and layer_type in _supported_layers and not disable_warning:
                print_fn(f'[warning] skip {name} with {num_params} parameters')
            continue

        start = time.time()
        # torch.cuda.synchronize()
        grad_per_sample = layer.weight.grad_ps
        grad = layer.weight.grad
        param_shape = grad.shape
        assert grad_per_sample is not None, "grad_ps not found, call compute_grad_ps(model) first"
        batch_size = grad_per_sample.shape[0]

        grad_per_sample = grad_per_sample.reshape(batch_size, -1)
        # outer product + average over batch
        hess = torch.einsum('ni,nj->ij', grad_per_sample, grad_per_sample) / batch_size

        # torch.cuda.synchronize()
        end = time.time()
        timeit_dict['hessian'] += end - start

        if quantum_sparsity_ratio < 1.0:
            hess = prune_matrix_for_quantum_sparsity(
                hess, quantum_sparsity=floor(quantum_sparsity_ratio * hess.shape[0]))

        # if profile_hessian:
        #     print_fn(f"[profile] min, max of the Hessian of {name}: {torch.min(hess).item()}, {torch.max(hess).item()}")

        if profile_hessian and "intermediate.dense" in name:
            ffn_layer_hessian_condition_numbers_dict[name] = torch.linalg.cond(hess).item()
            half_cum_percent_numel_per_row = torch.tensor(_compute_bacthed_vector_cum_percent_numel(hess, 0.5))
            statistics = (torch.min(half_cum_percent_numel_per_row).item(),
                          torch.median(half_cum_percent_numel_per_row).item(),
                          torch.mean(half_cum_percent_numel_per_row.float()).int().item(),
                          torch.max(half_cum_percent_numel_per_row).item())
            ffn_layer_hessian_half_percent_numel_dict[name] = statistics

        start = time.time()
        # torch.cuda.synchronize()
        # todo: -> max_singular_value * normalization_epsilon

        max_val = torch.max(torch.abs(hess)) + damping
        hess = hess + torch.eye(hess.shape[0], device=hess.device, dtype=hess.dtype) * max_val * normalization_epsilon

        # if name == "fc_layers.0":
        #     singulars = torch.linalg.svdvals(hess)
        #     singular_log = {
        #         "max_singular": singulars[0].item(),
        #         "min_singular": singulars[-1].item(),
        #         "condition_number": (singulars[0] / singulars[-1]).item()
        #     }
        #     wandb.log(singular_log)
        #     print_fn(f"[Profile] Singular values: max={singulars[0].item()} min={singulars[-1].item()} ")
        if num_params > 1000 and profile_hessian:
            singulars = torch.linalg.svdvals(hess)
            all_condition_numbers.append((singulars[0] / singulars[-1]).item())
            all_num_params.append(num_params)

        inv_hess = torch.inverse(hess)
        # torch.cuda.synchronize()
        end = time.time()
        timeit_dict['inversion'] += end - start

        start = time.time()
        # torch.cuda.synchronize()
        grad = grad.flatten()
        grad = torch.einsum('ij,j->i', inv_hess, grad)
        layer.weight.grad = grad.reshape(param_shape)
        # torch.cuda.synchronize()
        end = time.time()
        timeit_dict['rescale'] += end - start
        delattr(layer.weight, 'grad_ps')

        if layer.bias is not None:
            start = time.time()
            # torch.cuda.synchronize()
            grad_per_sample = layer.bias.grad_ps.reshape(batch_size, -1)
            grad = layer.bias.grad
            hess = torch.einsum('ni,nj->ij', grad_per_sample, grad_per_sample) / batch_size
            # torch.cuda.synchronize()
            end = time.time()
            timeit_dict['hessian'] += end - start

            start = time.time()
            # torch.cuda.synchronize()
            max_val = torch.max(torch.abs(hess)) + damping  # todo: -> max_singular_value * normalization_epsilon
            hess = hess + torch.eye(hess.shape[0], device=hess.device,
                                    dtype=hess.dtype) * max_val * normalization_epsilon
            inv_hess = torch.inverse(hess)
            # torch.cuda.synchronize()
            end = time.time()
            timeit_dict['inversion'] += end - start

            start = time.time()
            torch.cuda.synchronize()
            grad = torch.einsum('ij,j->i', inv_hess, grad)
            layer.bias.grad = grad
            torch.cuda.synchronize()
            end = time.time()
            timeit_dict['rescale'] += end - start
            delattr(layer.bias, 'grad_ps')

    if profile_hessian:
        wandb.log({
            "average_condition_number": sum(all_condition_numbers) / len(all_condition_numbers),
            "max_condition_number": max(all_condition_numbers),
            "min_condition_number": min(all_condition_numbers),
        })
    # print(all_condition_numbers) # fixme: remove it
    # print(all_num_params) # fixme: remove it
    # if profile_hessian:
    #     print_fn(f"[profile] Hessian condition numbers of FFN layers: {ffn_layer_hessian_condition_numbers_dict}")
    #     print_fn(f"[profile] Hessian half cum percent numel of FFN layers: {ffn_layer_hessian_half_percent_numel_dict}")

    return timeit_dict
