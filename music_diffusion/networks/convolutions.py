# -*- coding: utf-8 -*-
from typing import Literal

import torch as th
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .utils import ChannelModule


class _BaseConv(nn.Sequential, ChannelModule):
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


############
# Waveform #
############

# Unit modules


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


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
    ):
        super().__init__()
        self.__conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=0,
            )
        )

        self.__padding = dilation * (kernel_size - 1) + (1 - stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv(
            F.pad(x, (self.__padding, 0), mode="constant", value=0.0)
        )
        return out


# https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py
class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        dilation: int,
    ) -> None:
        super().__init__()

        self.__conv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            )
        )

        self.__padding = kernel_size - 1

    # pylint: disable=arguments-renamed
    def forward(self, x: Tensor) -> Tensor:
        out: th.Tensor = self.__conv(x)[..., : (x.size(2) * self.__padding)]
        return out


class StrideCausalConvTranspose1d(CausalConvTranspose1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            kernel_size // 4,
            0,
            dilation,
        )


# Blocks


class CausalConvBlock(_BaseConv):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        super().__init__(
            out_channels,
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=1,
                dilation=dilation,
            ),
            nn.Mish(),
        )


class CausalConvTransposeBlock(_BaseConv):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        super().__init__(
            out_channels,
            CausalConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=dilation,
            ),
            nn.Mish(),
        )


class StrideCausalConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ) -> None:
        conv_constructor = {
            "up": StrideCausalConvTranspose1d,
            "down": CausalConv1d,
        }

        super().__init__(
            out_channels,
            conv_constructor[scale](
                in_channels,
                out_channels,
                kernel_size=8,
                stride=4,
                dilation=1,
            ),
            nn.Mish(),
        )
