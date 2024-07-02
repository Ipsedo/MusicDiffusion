# -*- coding: utf-8 -*-
from typing import Literal

from torch.nn import functional as F

from ..convolutions import _BaseConv
from .activations import Hermite
from .convolutions import Conv2dKan, ConvTr2dKan


class OutChannelProj(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            Conv2dKan(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_fun=Hermite(5),
                res_act_fun=F.mish,
            ),
        )


class StrideConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ) -> None:
        conv_constructor = {
            "up": ConvTr2dKan,
            "down": Conv2dKan,
        }

        super().__init__(
            out_channels,
            conv_constructor[scale](
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                act_fun=Hermite(5),
                res_act_fun=F.mish,
            ),
        )


class ConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            Conv2dKan(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act_fun=Hermite(5),
                res_act_fun=F.mish,
            ),
        )
