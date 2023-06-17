# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch as th
from torch import nn

from .attention import SelfAttention2d
from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock
from .time import SinusoidTimeEmbedding, TimeBypass, TimeWrapper


class TimeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[Tuple[int, int]],
        use_attention: List[bool],
        attention_heads: int,
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

        encoder_attention = use_attention.copy()
        decoder_attention = reversed(use_attention)

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
                    ConvBlock(c_i, c_o),
                    SelfAttention2d(
                        c_o,
                        attention_heads,
                        c_o,
                        c_o // 4,
                        c_o // 4,
                    )
                    if use_att
                    else nn.Identity(),
                    ConvBlock(c_o, c_o),
                ),
            )
            for use_att, (c_i, c_o) in zip(
                encoder_attention, encoding_channels
            )
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
                    ConvBlock(c_i, c_i),
                    SelfAttention2d(
                        c_i,
                        attention_heads,
                        c_i,
                        c_i // 4,
                        c_i // 4,
                    )
                    if use_att
                    else nn.Identity(),
                    ConvBlock(c_i, c_o),
                )
            )
            for use_att, (c_i, c_o) in zip(
                decoder_attention, decoding_channels
            )
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
