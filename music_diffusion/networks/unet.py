# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .convolutions import ConvBlock, OutChannelProj, StrideConvBlock
from .recurrent import MiddleTransformer
from .tau import ConditionEncoder
from .time import (
    ConditionTimeBypass,
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
        trf_dim: int,
        trf_hidden_dim: int,
        trf_num_heads: int,
        tau_dim: int,
        tau_hidden_dim: int,
        tau_layers: int,
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

        # Time embedder
        self.__time_embedder = SinusoidTimeEmbedding(steps, time_size)

        # Tau - Input encoder
        self.__tau = ConditionEncoder(tau_dim, tau_hidden_dim, tau_layers)

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

        self.__middle_conv_1 = TimeWrapper(
            time_size,
            ConvBlock(c_m, c_m),
        )

        self.__lstm = ConditionTimeBypass(
            MiddleTransformer(
                c_m,
                trf_dim,
                trf_hidden_dim,
                tau_hidden_dim,
                trf_num_heads,
                512,
            )
        )

        self.__middle_conv_2 = TimeWrapper(
            time_size,
            ConvBlock(c_m, c_m),
        )

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
        x_t: th.Tensor,
        t: th.Tensor,
        y: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        time_vec = self.__time_embedder(t)

        bypasses = []

        y_encoded = self.__tau(y)

        out = x_t

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            out = block(out, time_vec)
            bypasses.append(out)
            out = down(out)

        out = self.__middle_conv_1(out, time_vec)
        out = self.__lstm(out, y_encoded)
        out = self.__middle_conv_2(out, time_vec)

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
