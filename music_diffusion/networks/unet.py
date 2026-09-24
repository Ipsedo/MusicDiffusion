from typing import List, Tuple

import torch as th
from torch import nn

from .convolutions import ConvBlock, OutChannelProj, StrideConvBlock
from .time import SequentialTimeWrapper, SinusoidTimeEmbedding, TimeBypass


class TimeUNet(nn.Module):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        group_norm_num: list[int],
        time_size: int,
        steps: int,
    ) -> None:
        super().__init__()

        assert all(
            channels[i][1] == channels[i + 1][0]
            for i in range(len(channels) - 1)
        )

        encoding_channels = channels.copy()
        encoding_group_norm_num = group_norm_num.copy()

        decoding_channels = [(c_o, c_i) for c_i, c_o in reversed(channels)]
        decoding_channels[-1] = (
            decoding_channels[-1][0],
            decoding_channels[-1][0],
        )
        decoding_group_norm_num = list(reversed(group_norm_num.copy()))

        self.__time_embedder = SinusoidTimeEmbedding(steps, time_size)

        # Encoder stuff

        self.__encoder = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    ConvBlock(c_i, c_o, g),
                    ConvBlock(c_o, c_o, g),
                ],
            )
            for (c_i, c_o), g in zip(
                encoding_channels, encoding_group_norm_num
            )
        )

        self.__encoder_down = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_o, c_o, g, "down"))
            for (_, c_o), g in zip(encoding_channels, encoding_group_norm_num)
        )

        # Middle stuff
        c_m = encoding_channels[-1][1]
        g_m = encoding_group_norm_num[-1]
        self.__middle_block = SequentialTimeWrapper(
            time_size,
            [
                ConvBlock(c_m, c_m, g_m),
                ConvBlock(c_m, c_m, g_m),
            ],
        )

        # Decoder stuff
        self.__decoder_up = nn.ModuleList(
            TimeBypass(StrideConvBlock(c_i, c_i, g, "up"))
            for (c_i, _), g in zip(decoding_channels, decoding_group_norm_num)
        )

        self.__decoder = nn.ModuleList(
            SequentialTimeWrapper(
                time_size,
                [
                    ConvBlock(c_i * 2, c_o, g),
                    ConvBlock(c_o, c_o, g),
                ],
            )
            for (c_i, c_o), g in zip(
                decoding_channels, decoding_group_norm_num
            )
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

        out = self.__middle_block(out, time_vec)

        for up, bypass, block in zip(
            self.__decoder_up,
            reversed(bypasses),
            self.__decoder,
        ):
            out = up(out)
            out = th.cat([out, bypass], dim=2)
            out = block(out, time_vec)

        print(out.size())
        eps: th.Tensor = self.__eps_end_conv(out)
        v: th.Tensor = self.__v_end_conv(out)

        return eps, v
