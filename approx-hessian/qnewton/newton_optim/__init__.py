# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/26
from .autograd_utils import (
    compute_grad_ps_,
    clear_backprops_,
    add_hooks_,
    remove_hooks_,
    disable_hooks_,
    enable_hooks_,
)
from .hessian import (
    compute_empirical_fisher_,
    inverse_fisher_,
    rescale_gradient_by_fisher_inv_,
    rescale_gradient_for_all_
)
