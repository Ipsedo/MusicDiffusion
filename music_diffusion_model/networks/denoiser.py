import torch as th
import torch.nn as nn

from .convolutions import ConvBlock
from .time import TimeWrapper


class ConvWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:

        super().__init__()

        self.__conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 5

        b, t, _, w, h = x.size()

        x = x.flatten(0, 1)

        out: th.Tensor = self.__conv(x)

        return out.view(b, t, -1, w, h)


class Denoiser(nn.Module):
    def __init__(
        self,
        channels: int,
        steps: int,
        time_size: int,
        beta_1: float,
        beta_t: float,
    ):
        super().__init__()

        self.__steps = steps

        self.__betas = th.linspace(beta_1, beta_t, steps=self.__steps)[
            None, :, None, None, None
        ].flip([1])

        self.__alphas = 1 - self.__betas
        self.__alpha_cumprod = th.cumprod(self.__alphas, dim=1)

        layers = [
            (channels, 8),
            (8, 16),
            (16, 32),
            (32, 16),
            (16, 8),
            (8, channels),
        ]

        self.__eps = nn.Sequential(
            TimeWrapper(
                ConvWrapper(layers[0][0] + time_size, layers[0][1]),
                steps,
                time_size,
            ),
            *[ConvWrapper(c_i, c_o) for c_i, c_o in layers[1:]]
        )

    def forward(self, x_noised: th.Tensor) -> th.Tensor:
        assert len(x_noised.size()) == 5
        assert x_noised.size(1) == self.__steps

        t = th.arange(self.__steps).flip([0])

        eps = self.__eps((x_noised, t))
        out: th.Tensor = (1.0 / th.sqrt(self.__alphas)) * (
            x_noised - eps * self.__betas / th.sqrt(1.0 - self.__alpha_cumprod)
        )

        return out
