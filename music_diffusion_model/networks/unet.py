from typing import List, Tuple

import torch as th
import torch.nn as nn
from torchvision.ops import Permute

from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock
from .time import TimeBypass, TimeEmbeder, TimeWrapper


class TimeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[Tuple[int, int]],
        time_size: int,
        steps: int,
    ) -> None:
        super().__init__()

        assert all(
            [
                hidden_channels[i][1] == hidden_channels[i + 1][0]
                for i in range(len(hidden_channels) - 1)
            ]
        )

        encoding_channels = hidden_channels.copy()
        decoding_channels = [
            (c_o, c_i) for c_i, c_o in reversed(hidden_channels)
        ]

        self.__time_embedder = TimeEmbeder(steps, time_size)

        # Encoder stuff

        self.__start_conv = TimeBypass(
            ConvBlock(
                in_channels,
                encoding_channels[0][0],
            )
        )

        self.__to_channels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(time_size, c_i),
                    Permute([0, 2, 1]),
                    nn.BatchNorm1d(c_i),
                    Permute([0, 2, 1]),
                )
                for c_i, _ in encoding_channels
            ]
        )

        self.__encoder = nn.ModuleList(
            [
                TimeWrapper(
                    nn.Sequential(
                        ConvBlock(c_i, c_o),
                        ConvBlock(c_o, c_o),
                    )
                )
                for c_i, c_o in encoding_channels
            ]
        )

        self.__encoder_down = nn.ModuleList(
            [
                TimeBypass(StrideConvBlock(c_o, c_o, "down"))
                for _, c_o in encoding_channels
            ]
        )

        # Decoder stuff

        self.__decoder = nn.ModuleList(
            [
                TimeBypass(
                    nn.Sequential(
                        ConvBlock(c_i, c_i),
                        ConvBlock(c_i, c_o),
                    )
                )
                for c_i, c_o in decoding_channels
            ]
        )

        self.__decoder_up = nn.ModuleList(
            [
                TimeBypass(StrideConvBlock(c_i, c_i, "up"))
                for c_i, _ in decoding_channels
            ]
        )

        self.__end_conv = TimeBypass(
            EndConvBlock(
                decoding_channels[-1][1],
                out_channels,
            )
        )

    def forward(self, img: th.Tensor, t: th.Tensor) -> th.Tensor:
        time_vec = self.__time_embedder(t)

        residuals = []

        out: th.Tensor = self.__start_conv(img)

        for to_channels, block, down in zip(
            self.__to_channels,
            self.__encoder,
            self.__encoder_down,
        ):
            res = block(out, to_channels(time_vec))
            residuals.append(res)

            out = down(res)

        for block, up, res in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(residuals),
        ):
            out_up = up(out)
            out = block(out_up + res)

        out = self.__end_conv(out)

        return out
