# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .convolutions import (
    CausalConvBlock,
    OutChannelProj1d,
    StrideCausalConvBlock,
)
from .liquid import LiquidRecurrentOutput
from .time import (
    SequentialTimeWrapper,
    SinusoidTimeEmbedding,
    TimeBypass,
    TimeWrapper,
)


class TimeUNet(nn.Module):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        time_size: int,
        steps: int,
        neuron_number: int,
    ) -> None:
        super().__init__()

        assert all(
            channels[i][1] == channels[i + 1][0]
            for i in range(len(channels) - 1)
        )

        encoding_channels = channels.copy()
        decoding_channels = [(c_o, c_i) for c_i, c_o in reversed(channels)]
        decoding_channels[-1] = (
            decoding_channels[-1][0],
            decoding_channels[-1][0],
        )

        self.__time_embedder = SinusoidTimeEmbedding(steps, time_size)

        # Encoder stuff

        self.__encoder = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    CausalConvBlock(c_i, c_o, 1),
                    CausalConvBlock(c_o, c_o, 2),
                    CausalConvBlock(c_o, c_o, 4),
                    CausalConvBlock(c_o, c_o, 8),
                    CausalConvBlock(c_o, c_o, 16),
                ],
            )
            for c_i, c_o in encoding_channels
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideCausalConvBlock(c_o, c_o, "down"))
            for _, c_o in encoding_channels
        )

        # Middle stuff
        c_m = encoding_channels[-1][1]
        self.__middle_block = TimeWrapper(
            time_size,
            LiquidRecurrentOutput(neuron_number, c_m, 6),
        )

        # Decoder stuff
        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideCausalConvBlock(c_i, c_i, "up"))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    CausalConvBlock(c_i * 2, c_i, 16),
                    CausalConvBlock(c_i, c_i, 8),
                    CausalConvBlock(c_i, c_i, 4),
                    CausalConvBlock(c_i, c_i, 2),
                    CausalConvBlock(c_i, c_o, 1),
                ],
            )
            for c_i, c_o in decoding_channels
        )

        c_o = decoding_channels[-1][1]
        out_channels = encoding_channels[0][0]
        self.__eps_end_conv = TimeBypass(
            OutChannelProj1d(c_o, out_channels),
        )

        self.__v_end_conv = TimeBypass(
            OutChannelProj1d(c_o, out_channels),
        )

    def forward(
        self, img: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        time_vec = self.__time_embedder(t)

        bypasses = []

        out = img

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            out = block(out, time_vec)
            bypasses.append(out)
            out = down(out)

        out = self.__middle_block(out, time_vec) + out

        for up, bypass, block in zip(
            self.__decoder_up,
            reversed(bypasses),
            self.__decoder,
        ):
            out = up(out)
            out = th.cat([out, bypass], dim=2)
            out = block(out, time_vec)

        eps: th.Tensor = self.__eps_end_conv(out)
        v: th.Tensor = self.__v_end_conv(out)

        return eps, v
