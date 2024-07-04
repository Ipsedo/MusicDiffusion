# -*- coding: utf-8 -*-
from typing import TypeVar

from torch import nn
from torch.nn.utils.parametrizations import weight_norm

M = TypeVar("M", bound=nn.Module)


def kan_weight_norm(m: M) -> M:
    parametrized_m: M = weight_norm(m, name="_c", dim=1)
    return parametrized_m
