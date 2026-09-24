from typing import Literal

import torch as th
from torch import nn

from .time import TimeToScaleShift


class OutChannelProj(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )


class StrideConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm_num: int,
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
                bias=False,
            ),
            nn.GroupNorm(group_norm_num, out_channels),
            nn.SiLU(),
        )


class TimeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm_num: int,
        time_size: int,
    ) -> None:
        super().__init__()

        self.__conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(group_norm_num, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(group_norm_num, out_channels),
        )

        self.__time_decoder = TimeToScaleShift(out_channels, time_size)

        self.__act = nn.SiLU()

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        b, t = x.size()[:2]

        scale, shift = self.__time_decoder(time_emb)

        out: th.Tensor = self.__conv(x.flatten(0, 1))
        out = th.unflatten(out, 0, (b, t))

        out = out * (scale + 1.0) + shift

        out = self.__act(out)

        return out
