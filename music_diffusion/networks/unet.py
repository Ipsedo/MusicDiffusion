from typing import List, Tuple

import torch as th
from torch import nn

from .attention import SelfAttention2d
from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock
from .time import TimeBypass, TimeEmbeder, TimeWrapper


class TimeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[Tuple[int, int]],
        use_attentions: List[bool],
        attention_heads: int,
        time_size: int,
        steps: int,
        norm_groups: int,
    ) -> None:
        super().__init__()

        assert len(hidden_channels) == len(use_attentions)
        assert all(
            hidden_channels[i][1] == hidden_channels[i + 1][0]
            for i in range(len(hidden_channels) - 1)
        )

        encoding_channels = hidden_channels.copy()
        decoding_channels = [
            (c_o, c_i) for c_i, c_o in reversed(hidden_channels)
        ]

        encoder_attentions = use_attentions.copy()
        decoder_attentions = reversed(use_attentions)

        self.__time_embedder = TimeEmbeder(steps, time_size)

        # Encoder stuff

        self.__start_conv = TimeBypass(
            ConvBlock(
                in_channels,
                encoding_channels[0][0],
                norm_groups,
            )
        )

        self.__encoder = nn.ModuleList(
            TimeWrapper(
                time_size,
                c_i,
                nn.Sequential(
                    ConvBlock(c_i, c_o, norm_groups),
                    SelfAttention2d(
                        c_o,
                        attention_heads,
                        c_o,
                        c_o // 8,
                        c_o // 8,
                    )
                    if use_att
                    else nn.Identity(),
                    ConvBlock(c_o, c_o, norm_groups),
                ),
                norm_groups,
            )
            for use_att, (c_i, c_o) in zip(
                encoder_attentions, encoding_channels
            )
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, "down", norm_groups))
            for _, c_o in encoding_channels
        )

        # Middle stuff

        c_m = encoding_channels[-1][1]
        self.__middle_block = TimeBypass(
            nn.Sequential(
                ConvBlock(c_m, c_m, norm_groups),
                ConvBlock(c_m, c_m, norm_groups),
            )
        )

        # Decoder stuff

        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, "up", norm_groups))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            TimeWrapper(
                time_size,
                c_i * 2,
                nn.Sequential(
                    ConvBlock(c_i * 2, c_i, norm_groups),
                    SelfAttention2d(
                        c_i,
                        attention_heads,
                        c_i,
                        c_i // 8,
                        c_i // 8,
                    )
                    if use_att
                    else nn.Identity(),
                    ConvBlock(c_i, c_o, norm_groups),
                ),
                norm_groups,
            )
            for use_att, (c_i, c_o) in zip(
                decoder_attentions, decoding_channels
            )
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

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            out = block(out, time_vec)
            residuals.append(out)
            out = down(out)

        out = self.__middle_block(out)

        for block, up, res in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(residuals),
        ):
            out = up(out)
            out = th.cat([out, res], dim=2)
            out = block(out, time_vec)

        out = self.__end_conv(out)

        return out
