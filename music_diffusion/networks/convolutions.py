# -*- coding: utf-8 -*-
from typing import Literal

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class _BaseConv(nn.Sequential):
    def __init__(self, out_channels: int, *modules: nn.Module):
        super().__init__(*modules)

        self.__out_channels = out_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels


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


# Waveform


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
        )

        self.__padding = (kernel_size - 1) * dilation

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(
            F.pad(x, (self.__padding, 0), mode="constant", value=0.0)
        )


class CausalConvBlock(_BaseConv):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=dilation,
                )
            ),
            nn.Mish(),
        )


class StrideConv1dBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ) -> None:
        conv_constructor = {
            "up": nn.ConvTranspose1d,
            "down": nn.Conv1d,
        }

        super().__init__(
            out_channels,
            weight_norm(
                conv_constructor[scale](
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.Mish(),
        )


class OutChannelProj1d(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            out_channels,
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            ),
        )
