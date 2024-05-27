# -*- coding: utf-8 -*-
from typing import Literal

from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .utils import _BaseConv


class ChannelProjBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                )
            ),
            nn.Mish(),
        )


class OutChannelProj(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                )
            ),
        )


class EndConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                )
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
            "up": nn.ConvTranspose2d,
            "down": nn.Conv2d,
        }

        super().__init__(
            out_channels,
            weight_norm(
                conv_constructor[scale](
                    in_channels,
                    out_channels,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            ),
            nn.Mish(),
        )


class ConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                )
            ),
            nn.Mish(),
        )
