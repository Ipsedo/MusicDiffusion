from abc import ABC, abstractmethod

import torch.nn as nn

from .normalization import PixelNorm


class AbstractConv(ABC, nn.Sequential):
    @property
    @abstractmethod
    def scale_factor(self) -> float:
        pass


class ConvEndBlock(AbstractConv):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.Tanh(),
        )

    @property
    def scale_factor(self) -> float:
        return 1.0


class StrideConvBlock(AbstractConv):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.GELU(),
            PixelNorm(),
        )

    @property
    def scale_factor(self) -> float:
        return 0.5


class StrideConvTrBlock(AbstractConv):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(0, 0),
            ),
            nn.GELU(),
            PixelNorm(),
        )

    @property
    def scale_factor(self) -> float:
        return 2.0


class StrideEndConvTrBlock(AbstractConv):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(0, 0),
            ),
            nn.Tanh(),
        )

    @property
    def scale_factor(self) -> float:
        return 2.0


class ConvBlock(AbstractConv):
    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: float
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.GELU(),
            nn.Upsample(
                scale_factor=scale_factor,
                mode="area",
            ),
        )

        self.__scale_factor = scale_factor

    @property
    def scale_factor(self) -> float:
        return self.__scale_factor
