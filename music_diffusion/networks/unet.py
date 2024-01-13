# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .attention import CrossAttention
from .convolutions import ConvBlock, OutChannelProj, StrideConvBlock
from .time import (
    ConditionTimeBypass,
    SequentialTimeWrapper,
    SinusoidTimeEmbedding,
    TimeBypass,
)


class TimeUNet(nn.Module):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        time_size: int,
        steps: int,
        condition_dim: int,
        trf_hidden_dim: int,
        trf_num_heads: int,
        trf_layers: int,
        kv_dim: int,
        kv_length: int,
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
                    ConvBlock(c_i, c_o),
                    ConvBlock(c_o, c_o),
                ],
            )
            for c_i, c_o in encoding_channels
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, "down"))
            for _, c_o in encoding_channels
        )

        # Middle stuff
        c_m = encoding_channels[-1][1]
        self.__middle_block = SequentialTimeWrapper(
            time_size,
            [
                ConvBlock(c_m, c_m),
                ConvBlock(c_m, c_m),
            ],
        )

        # pylint: disable=duplicate-code
        self.__cross_attention = ConditionTimeBypass(
            CrossAttention(
                c_m,
                condition_dim,
                trf_hidden_dim,
                trf_num_heads,
                trf_layers,
                kv_dim,
                kv_length,
            )
        )
        # pylint: enable=duplicate-code

        # Decoder stuff
        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, "up"))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    ConvBlock(c_i * 2, c_o),
                    ConvBlock(c_o, c_o),
                ],
            )
            for c_i, c_o in decoding_channels
        )

        c_o = decoding_channels[-1][1]
        out_channels = encoding_channels[0][0]
        self.__eps_end_conv = TimeBypass(
            OutChannelProj(c_o, out_channels),
        )

        self.__v_end_conv = TimeBypass(
            OutChannelProj(c_o, out_channels),
        )

    def forward(
        self,
        img: th.Tensor,
        t: th.Tensor,
        y: th.Tensor,
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

        out = self.__middle_block(out, time_vec)
        out = self.__cross_attention(out, y)

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
