# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .kan import ConvBlock, OutChannelProj, StrideConvBlock
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

        # Diffusion step embedding
        self.__time_embedder = SinusoidTimeEmbedding(steps, time_size)

        # Encoder stuff
        self.__encoder_down = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    ConvBlock(c_i, c_o),
                    StrideConvBlock(c_o, c_o, "down"),
                ],
            )
            for c_i, c_o in encoding_channels
        )

        # Middle stuff
        c_m = encoding_channels[-1][1]
        self.__middle_block = TimeWrapper(time_size, ConvBlock(c_m, c_m))

        # Decoder stuff
        self.__decoder_up = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    StrideConvBlock(c_i, c_i, "up"),
                    ConvBlock(c_i, c_o),
                ],
            )
            for c_i, c_o in decoding_channels
        )

        # Output stuff
        c_o = decoding_channels[-1][1]
        out_channels = encoding_channels[0][0]
        self.__eps_end_conv = TimeBypass(OutChannelProj(c_o, out_channels))
        self.__v_end_conv = TimeBypass(OutChannelProj(c_o, out_channels))

    def forward(
        self, img: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        time_vec = self.__time_embedder(t)

        bypasses = []

        out = img

        for down in self.__encoder_down:
            out = down(out, time_vec)
            bypasses.append(out)

        out = self.__middle_block(out, time_vec)

        for up, bypass in zip(
            self.__decoder_up,
            reversed(bypasses),
        ):
            out = up(out + bypass, time_vec)

        eps: th.Tensor = self.__eps_end_conv(out)
        v: th.Tensor = self.__v_end_conv(out)

        return eps, v
