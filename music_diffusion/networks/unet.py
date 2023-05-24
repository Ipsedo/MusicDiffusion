# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock
from .time import SinusoidTimeEmbedding, TimeBypass, TimeWrapper


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
            hidden_channels[i][1] == hidden_channels[i + 1][0]
            for i in range(len(hidden_channels) - 1)
        )

        encoding_channels = hidden_channels.copy()
        decoding_channels = [
            (c_o, c_i) for c_i, c_o in reversed(hidden_channels)
        ]

        self.__time_embedder = SinusoidTimeEmbedding(steps, time_size)

        # Encoder stuff

        self.__start_conv = TimeBypass(
            ConvBlock(
                in_channels,
                encoding_channels[0][0],
            )
        )

        self.__encoder = nn.ModuleList(
            TimeWrapper(
                time_size,
                c_i,
                nn.Sequential(
                    ConvBlock(c_i, c_i),
                    ConvBlock(c_i, c_o),
                ),
            )
            for c_i, c_o in encoding_channels
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, "down"))
            for _, c_o in encoding_channels
        )

        # Middle stuff

        c_m = encoding_channels[-1][1]
        self.__middle_block = TimeBypass(
            nn.Sequential(
                ConvBlock(c_m, c_m),
                ConvBlock(c_m, c_m),
            )
        )

        # Decoder stuff

        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, "up"))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            TimeBypass(
                nn.Sequential(
                    ConvBlock(c_i, c_o),
                    ConvBlock(c_o, c_o),
                )
            )
            for c_i, c_o in decoding_channels
        )

        self.__eps_end_conv = TimeBypass(
            EndConvBlock(
                decoding_channels[-1][1],
                out_channels,
            )
        )

    def forward(self, img: th.Tensor, t: th.Tensor) -> th.Tensor:
        time_vec = self.__time_embedder(t)

        bypasses = []

        out: th.Tensor = self.__start_conv(img)

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            out = block(out, time_vec)
            bypasses.append(out)
            out = down(out)

        out = self.__middle_block(out)

        for block, up, bypass in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(bypasses),
        ):
            out = up(out)
            out = out + bypass
            out = block(out)

        eps: th.Tensor = self.__eps_end_conv(out)

        return eps
