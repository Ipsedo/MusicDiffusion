# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class LogLU(nn.Module):
    def __init__(self, epsilon: float = 1e-12):
        super().__init__()
        self.__eps = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.where(x < 0, -th.log(1.0 - x + self.__eps), x)
