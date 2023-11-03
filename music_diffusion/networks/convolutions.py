# -*- coding: utf-8 -*-
from typing import Literal

import torch as th
from torch import nn

from .normalization import PixelNorm
from .time import TimeBypass, TimeWrapper


class ChannelProjBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.Mish(),
            PixelNorm(),
        )


class OutChannelProjBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
        )


class EndConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )


class StrideConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ) -> None:
        conv_constructor = {
            "up": nn.ConvTranspose2d,
            "down": nn.Conv2d,
        }

        super().__init__(
            conv_constructor[scale](
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Mish(),
            PixelNorm(),
        )


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.Mish(),
            PixelNorm(),
        )


class DoubleConvBlock(nn.Module):
    def __init__(self, c_i: int, c_m: int, c_o: int, time_size: int) -> None:
        super().__init__()

        self.__conv_1 = TimeWrapper(time_size, c_m, ConvBlock(c_i, c_m))
        self.__conv_2 = TimeBypass(ConvBlock(c_m, c_o))

    def forward(self, x: th.Tensor, time_vec: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv_1(x, time_vec)
        out = self.__conv_2(out)
        return out
