# -*- coding: utf-8 -*-
from abc import ABC
from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_

from .activations import ActivationFunction

###############
# Convolution #
###############


# pylint: disable=too-many-instance-attributes
class _AbstractConv2dKan(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self.__w_b = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1, 1)
        )

        self.__w_s = nn.Parameter(
            th.ones(in_channels, out_channels, kernel_size * kernel_size, 1, 1)
        )

        self.__c = nn.Parameter(
            th.ones(
                in_channels,
                out_channels,
                kernel_size * kernel_size,
                1,
                self.__act_fun.get_size(),
            )
        )

        xavier_normal_(self.__w_b, 1e-3)
        normal_(self.__c, 0, 1e-3)

        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def _activation(self, flattened_x: th.Tensor) -> th.Tensor:
        return th.sum(
            self.__w_b * self.__res_act_fun(flattened_x).unsqueeze(-1)
            + self.__w_s * self.__c * self.__act_fun(flattened_x),
            dim=[1, 5],  # sum over input and activation spaces
        )


# pylint: disable=too-many-instance-attributes
class Conv2dKan(_AbstractConv2dKan):
    def __get_output_size(self, size: int) -> int:
        return (
            size - self._kernel_size + 2 * self._padding
        ) // self._stride + 1

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size(1) == self._in_channels

        b, c, h, w = x.size()

        output_height = self.__get_output_size(h)
        output_width = self.__get_output_size(w)

        return th.sum(
            self._activation(
                F.unfold(
                    x,
                    self._kernel_size,
                    1,
                    self._padding,
                    self._stride,
                ).view(b, c, 1, self._kernel_size**2, -1)
            ),
            dim=2,  # sum over window : dim=2
        ).view(b, -1, output_height, output_width)


##############
# Transposed #
##############


# pylint: disable=too-many-instance-attributes
class ConvTr2dKan(_AbstractConv2dKan):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            act_fun,
            res_act_fun,
        )
        self.__out_channels = out_channels

    def __get_output_size(self, size: int) -> int:
        return (
            self._stride * (size - 1) + self._kernel_size - 2 * self._padding
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4
        assert x.size(1) == self._in_channels

        b, _, h, w = x.size()

        output_height = self.__get_output_size(h)
        output_width = self.__get_output_size(w)

        return F.fold(
            self._activation(x.view(b, self._in_channels, 1, 1, h * w)).view(
                b, self.__out_channels * self._kernel_size**2, -1
            ),
            (output_height, output_width),
            self._kernel_size,
            1,
            self._padding,
            self._stride,
        )
