from abc import ABC, abstractmethod
from typing import Literal

import torch.nn as nn


class AbstractConv(ABC, nn.Sequential):
    @property
    @abstractmethod
    def scale_factor(self) -> float:
        pass


class ConvEndBlock(AbstractConv):
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
            nn.Tanh(),
        )

    @property
    def scale_factor(self) -> float:
        return 1.0


class StrideConv(AbstractConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ):
        conv_constructor = nn.Conv2d if scale == "down" else nn.ConvTranspose2d
        self.__scale_factor = 0.5 if scale == "down" else 2.0

        super().__init__(
            conv_constructor(
                in_channels,
                out_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.GELU(),
        )

    @property
    def scale_factor(self) -> float:
        return self.__scale_factor


class ConvBlock(AbstractConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
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
