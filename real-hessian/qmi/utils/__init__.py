# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/7/10
"""
Seed all
"""
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
