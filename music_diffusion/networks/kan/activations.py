# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from math import factorial

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


##########
# Spline #
##########


def b_spline(
    x: th.Tensor, k: int, n: int, x_min: float = 0.0, x_max: float = 1.0
) -> th.Tensor:
    x = x.unsqueeze(-1)

    def __knots(_i: th.Tensor) -> th.Tensor:
        return _i / n * (x_max - x_min) + x_min

    i_s = th.arange(-k, n, device=x.device)

    def __b_spline(curr_i_s: th.Tensor, curr_k: int) -> th.Tensor:
        if curr_k == 0:
            return th.logical_and(
                th.le(__knots(curr_i_s), x), th.lt(x, __knots(curr_i_s + 1))
            ).to(th.float)

        return __b_spline(curr_i_s, curr_k - 1) * (x - __knots(curr_i_s)) / (
            __knots(curr_i_s + curr_k) - __knots(curr_i_s)
        ) + __b_spline(curr_i_s + 1, curr_k - 1) * (
            __knots(curr_i_s + curr_k + 1) - x
        ) / (
            __knots(curr_i_s + curr_k + 1) - __knots(curr_i_s + 1)
        )

    return th.movedim(__b_spline(i_s, k), -1, 1)


class BSpline(ActivationFunction):
    def __init__(self, degree: int, grid_size: int) -> None:
        super().__init__()

        self.__degree = degree
        self.__grid_size = grid_size

    def forward(self, x: th.Tensor) -> th.Tensor:
        return b_spline(x, self.__degree, self.__grid_size)

    def get_size(self) -> int:
        return self.__grid_size + self.__degree


###########
# Hermite #
###########


def scale_hermite(x_hermite: th.Tensor) -> th.Tensor:
    return th.exp(-(x_hermite**2) / 2)


def hermite(x: th.Tensor, n: int) -> th.Tensor:
    h_s = [th.ones(*x.size(), device=x.device), 2 * x]

    # Iteratively compute Hermite polynomials from 2 to n
    for k in range(2, n + 1):
        h_s.append(2 * x * h_s[k - 1] - 2 * (k - 1) * h_s[k - 2])

    return th.slice_copy(th.stack(h_s, dim=1), 1, 1)


def hermite_coef(n: int) -> th.Tensor:
    coef: th.Tensor = th.zeros(n + 1, n)
    for i in range(n):
        for k in range((i + 1) // 2 + 1):
            coef[i + 1 - 2 * k, i] = (
                (-1) ** k
                / 2**k
                / factorial(k)
                / factorial(i + 1 - 2 * k)
                * th.exp(th.lgamma(th.tensor(i + 2)) / 2)
            )

    return coef


class Hermite(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n
        self._coef: th.Tensor
        self.register_buffer("_coef", hermite_coef(n))

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = x.unsqueeze(-1).expand(*x.size(), self.__n + 1)
        out = th.pow(out, th.arange(0, self.__n + 1, device=x.device))
        out = th.einsum("...a,ab->...b", out, self._coef)
        return out.movedim(-1, 1)

    def get_size(self) -> int:
        return self.__n


#########
# Tche
##########


def tcheb_coef(n: int) -> th.Tensor:
    coef = th.zeros(n + 1, n)
    for i in range(n):
        for k in range((i + 1) // 2 + 1):
            coef[i + 1 - 2 * k, i] = (
                (i + 1)
                / 2
                * (-1) ** k
                * 2 ** (i + 1 - 2 * k)
                * factorial(i - k)
                / factorial(k)
                / factorial(i + 1 - 2 * k)
            )

    return coef


class Tchebychev(ActivationFunction):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.__n = n
        self._coef: th.Tensor
        self.register_buffer("_coef", tcheb_coef(n))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.einsum(
            "...a,ab->...b",
            th.pow(
                th.tanh(x.unsqueeze(-1).expand(*x.size(), self.__n + 1)),
                th.arange(0, self.__n + 1, device=x.device),
            ),
            self._coef,
        ).movedim(-1, 1)

    def get_size(self) -> int:
        return self.__n
