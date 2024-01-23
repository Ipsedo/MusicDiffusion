# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class View(nn.Module):
    def __init__(self, *args: int) -> None:
        super().__init__()

        self.__shape = args

    def forward(self, x: th.Tensor) -> th.Tensor:
        b = x.size(0)
        return x.view(b, *self.__shape)


class Permute(nn.Module):
    def __init__(self, *args: int) -> None:
        super().__init__()
        self.__permutations = args

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.permute(*self.__permutations)
