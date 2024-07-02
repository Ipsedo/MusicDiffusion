# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import torch as th
from torch import nn

#############
# Functions #
#############


class ActivationFunction(ABC, nn.Module):
    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: th.Tensor) -> th.Tensor:
        pass


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), x]

    for i in range(1, n):
        h_s.append(x * h_s[i] - i * h_s[i - 1])

    return th.slice_copy(th.stack(h_s, dim=-1), -1, 1) / th.exp(
        th.lgamma(th.arange(2, n + 2, device=x.device)) / 2
    )


class Hermite(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n

    def forward(self, x: th.Tensor) -> th.Tensor:
        return hermite(x, self.__n)

    def get_size(self) -> int:
        return self.__n
