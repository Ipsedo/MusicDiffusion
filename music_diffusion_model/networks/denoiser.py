from typing import Callable

import torch as th
import torch.nn as nn

from .convolutions import ConvBlock, ConvEndBlock
from .time import TimeWrapper


class ConvWrapper(nn.Module):
    def __init__(self, conv: Callable[[th.Tensor], th.Tensor]) -> None:

        super().__init__()

        self.__conv = conv

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
        ]

        self.__alphas = 1.0 - self.__betas
        self.__alpha_cumprod = th.cumprod(self.__alphas, dim=1).flip([1])
        self.__alphas = self.__alphas.flip([1])

        self.__channels = channels

        layers = [
            (self.__channels, 8),
            (8, 16),
            (16, 32),
            (32, 16),
            (16, 8),
            (8, self.__channels),
        ]

        self.__eps = nn.Sequential(
            TimeWrapper(
                ConvWrapper(ConvBlock(layers[0][0] + time_size, layers[0][1])),
                steps,
                time_size,
            ),
            *[ConvWrapper(ConvBlock(c_i, c_o)) for c_i, c_o in layers[1:1]],
            ConvWrapper(ConvEndBlock(layers[-1][0], layers[-1][1]))
        )

    def forward(self, x_t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 5
        assert x_t.size(1) == self.__steps

        t = th.arange(self.__steps).flip([0])

        eps = self.__eps((x_t, t))
        out: th.Tensor = (1.0 / th.sqrt(self.__alphas)) * (
            x_t - eps * self.__betas / th.sqrt(1.0 - self.__alpha_cumprod)
        )

        return out

    def sample(self, x_t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        for t in reversed(range(0, self.__steps)):
            z = th.randn_like(x_t) if t > 1 else th.zeros_like(x_t)
            eps = self.__eps((x_t.unsqueeze(1), th.tensor([t]))).squeeze(1)
            x_t = (1.0 / th.sqrt(self.__alphas[:, t])) * (
                x_t
                - eps
                * (1.0 - self.__alphas[:, t])
                / th.sqrt(1.0 - self.__alpha_cumprod[:, t])
            ) + self.__betas[:, t] * z

        return x_t
