from typing import Literal

from torch import nn


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
        group_norm_num: int,
    ) -> None:
        super().__init__(
            out_channels,
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.GroupNorm(group_norm_num, out_channels),
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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )


class StrideConvBlock(_BaseConv):
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
            out_channels,
            conv_constructor[scale](
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.GroupNorm(group_norm_num, out_channels),
            nn.Mish(),
        )


class ConvBlock(_BaseConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_norm_num: int,
    ) -> None:
        super().__init__(
            out_channels,
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.GroupNorm(group_norm_num, out_channels),
            nn.Mish(),
        )
