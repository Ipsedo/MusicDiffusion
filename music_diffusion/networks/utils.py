# -*- coding: utf-8 -*-
from typing import Callable, Dict, Literal

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


class Agg(nn.Module):
    def __init__(
        self, method: Literal["mean", "sum", "max"], dim: int
    ) -> None:
        super().__init__()

        functions: Dict[str, Callable[..., th.Tensor]] = {
            "max": th.amax,
            "sum": th.sum,
            "mean": th.mean,
        }

        self.__fun = functions[method]
        self.__dim = dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__fun(x, dim=self.__dim)
