# -*- coding: utf-8 -*-
# @Author: Copied from https://github.com/cybertronai/autograd-hacks/blob/master/autograd_hacks.py
# @Time: 2023/11/27
"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""

from typing import List

import torch
import torch.nn as nn

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
_hooks_disabled: bool = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False  # global switch to catch double backprop errors on Hessian computation


def add_hooks_(model: nn.Module) -> None:
    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks_(model: nn.Module) -> None:
    # assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def disable_hooks_() -> None:
    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks_() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer,
                           'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach())


def clear_backprops_(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def compute_grad_ps_(
        model: nn.Module,
        loss_type: str = 'mean',
        remove_activations: bool = False,
        remove_backprops: bool = False,
) -> None:
    """
    Compute per-example gradients and save them under 'param.grad_ps'. Must be called after loss.backprop()
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        act_mat = layer.activations
        n = act_mat.shape[0]
        if loss_type == 'mean':
            grad_mat = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            grad_mat = layer.backprops_list[0]

        if layer_type == 'Linear' and len(act_mat.shape) == 3:
            # A is of shape [n, s, di], B is of shape [n, s, do]
            # we want to compute the [n, s, di, do] tensor of outer products and average over `s`
            # and the result is [n, di, do]
            act_mat = act_mat.reshape(n, -1, act_mat.shape[-1])
            grad_mat = grad_mat.reshape(n, -1, grad_mat.shape[-1])
            setattr(layer.weight, 'grad_ps', torch.einsum('nsd,nsh->ndh', act_mat, grad_mat))

            if layer.bias is not None:
                # grad_mat is of shape [n, s, do]
                setattr(layer.bias, 'grad_ps', grad_mat.sum(dim=1))

        elif layer_type == 'Linear' and len(act_mat.shape) == 2:
            setattr(layer.weight, 'grad_ps', torch.einsum('ni,nj->nij', grad_mat, act_mat))
            if layer.bias is not None:
                setattr(layer.bias, 'grad_ps', grad_mat)

        elif layer_type == 'Conv2d':
            act_mat = torch.nn.functional.unfold(act_mat,
                                                 kernel_size=layer.kernel_size,
                                                 padding=layer.padding,
                                                 stride=layer.stride)
            grad_mat = grad_mat.reshape(n, -1, act_mat.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', grad_mat, act_mat)
            shape = [n] + list(layer.weight.shape)
            setattr(layer.weight, 'grad_ps', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad_ps', torch.sum(grad_mat, dim=2))


        if remove_activations:
            delattr(layer, 'activations')
        if remove_backprops:
            delattr(layer, 'backprops_list')
