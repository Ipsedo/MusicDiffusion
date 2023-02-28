from typing import List, Tuple

import torch as th
import torch.nn as nn

from .attention import SelfAttention2d
from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock
from .time import TimeBypass, TimeConvBlock, TimeEmbeder


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        time_size: int,
        num_heads: int,
        use_att: bool,
    ) -> None:
        super().__init__()

        self.__conv_1 = TimeConvBlock(
            in_channels,
            hidden_channels,
            time_size,
        )

        self.__attention = (
            TimeBypass(
                SelfAttention2d(
                    hidden_channels,
                    num_heads,
                    hidden_channels,
                    hidden_channels // 2,
                    hidden_channels // 2,
                )
            )
            if use_att
            else nn.Identity()
        )

        self.__conv_2 = TimeConvBlock(
            hidden_channels,
            out_channels,
            time_size,
        )

    def forward(self, x: th.Tensor, tim_emb: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv_1(x, tim_emb)
        out = self.__attention(out)
        out = self.__conv_2(out, tim_emb)
        return out


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
            )
        )

        self.__encoder = nn.ModuleList(
            UNetBlock(
                c_i,
                c_o,
                c_o,
                time_size,
                attention_heads,
                use_att,
            )
            for use_att, (c_i, c_o) in zip(
                encoder_attentions, encoding_channels
            )
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, "down"))
            for _, c_o in encoding_channels
        )

        # Decoder stuff

        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, "up"))
            for c_i, _ in decoding_channels
        )

        self.__decoder = nn.ModuleList(
            UNetBlock(
                c_i,
                c_i,
                c_o,
                time_size,
                attention_heads,
                use_att,
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
            res = block(out, time_vec)
            residuals.append(res)

            out = down(res)

        for block, up, res in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(residuals),
        ):
            out_up = up(out)
            out = block(out_up + res, time_vec)

        out = self.__end_conv(out)

        return out
