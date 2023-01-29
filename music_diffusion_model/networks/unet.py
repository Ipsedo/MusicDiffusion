from typing import List, Tuple

import torch as th
import torch.nn as nn

from .convolutions import ConvBlock, EndConvBlock, StrideConvBlock


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoding_layers: List[Tuple[int, int]],
        decoding_layers: List[Tuple[int, int]],
    ) -> None:
        super().__init__()

        assert len(encoding_layers) == len(decoding_layers)

        self.__start_conv = ConvBlock(
            in_channels,
            encoding_layers[0][0],
        )

        self.__encoder = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(c_i, c_o),
                    ConvBlock(c_o, c_o),
                )
                for c_i, c_o in encoding_layers
            ]
        )

        self.__encoder_down = nn.ModuleList(
            [StrideConvBlock(c_o, c_o, "down") for _, c_o in encoding_layers]
        )

        self.__decoder = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(c_i, c_i),
                    ConvBlock(c_i, c_o),
                )
                for c_i, c_o in decoding_layers
            ]
        )

        self.__decoder_up = nn.ModuleList(
            [StrideConvBlock(c_i, c_i, "up") for c_i, _ in decoding_layers]
        )

        self.__end_conv = EndConvBlock(
            decoding_layers[-1][1],
            out_channels,
        )

    def forward(self, img: th.Tensor) -> th.Tensor:
        assert len(img.size()) == 4

        residuals = []

        out: th.Tensor = self.__start_conv(img)

        for block, down in zip(
            self.__encoder,
            self.__encoder_down,
        ):
            res = block(out)
            residuals.append(res)
            out = down(res)

        for block, up, res in zip(
            self.__decoder,
            self.__decoder_up,
            reversed(residuals),
        ):
            out_up = up(out)
            out = block(out_up + res)

        out = self.__end_conv(out)

        return out
