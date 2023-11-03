# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .convolutions import DoubleConvBlock, EndConvBlock, StrideConvBlock
from .time import SinusoidTimeEmbedding, TimeBypass


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
        c_i = encoding_channels[0][0]
        self.__start_conv = DoubleConvBlock(in_channels, c_i, c_i, time_size)

        self.__encoder = nn.ModuleList(
            DoubleConvBlock(c_i, c_o, c_o, time_size)
            for c_i, c_o in encoding_channels
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, "down"))
            for _, c_o in encoding_channels
        )

        # Middle stuff
        c_m = encoding_channels[-1][1]
        self.__middle_block = DoubleConvBlock(c_m, c_m, c_m, time_size)

        # Decoder stuff
        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, "up"))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            DoubleConvBlock(c_i * 2, c_i, c_o, time_size)
            for c_i, c_o in decoding_channels
        )

        c_o = decoding_channels[-1][1]
        self.__eps_end_conv = TimeBypass(
            EndConvBlock(c_o, out_channels),
        )

        self.__v_end_conv = TimeBypass(
            EndConvBlock(c_o, out_channels),
        )

    def forward(
        self, img: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        time_vec = self.__time_embedder(t)

        bypasses = []

        out: th.Tensor = self.__start_conv(img, time_vec)

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            out = block(out, time_vec)
            bypasses.append(out)
            out = down(out)

        out = self.__middle_block(out, time_vec)

        for block, up, bypass in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(bypasses),
        ):
            out = up(out)
            out = th.cat([out, bypass], dim=2)
            out = block(out, time_vec)

        eps: th.Tensor = self.__eps_end_conv(out)
        v: th.Tensor = self.__v_end_conv(out)

        return eps, v
