import torch as th
import torch.nn as nn

from .convolutions import AbstractConv, ConvBlock, ConvEndBlock
from .time import TimeWrapper


class ConvWrapper(nn.Module):
    def __init__(self, conv: AbstractConv) -> None:

        super().__init__()

        self.__conv = conv

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 5
        b, t, _, w, h = x.size()

        x = x.flatten(0, 1)
        out: th.Tensor = self.__conv(x)

        new_w, new_h = int(w * self.__conv.scale_factor), int(
            h * self.__conv.scale_factor
        )

        return out.view(b, t, -1, new_w, new_h)


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

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)[
            None, :, None, None, None
        ]
        alphas = 1.0 - betas
        alpha_cumprod = th.cumprod(alphas, dim=1)

        alpha_cumprod = alpha_cumprod.flip([1])
        alphas = alphas.flip([1])
        betas = betas.flip([1])

        self.alphas: th.Tensor
        self.alpha_cumprod: th.Tensor
        self.betas: th.Tensor

        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("betas", betas)

        self.__channels = channels

        layers_conv = [
            (self.__channels + time_size, 16),
            (16, 32),
        ]

        layer_conv_tr = [
            (32, 16),
            (16, 8),
        ]

        self.__eps = nn.Sequential(
            TimeWrapper(
                ConvWrapper(
                    ConvBlock(layers_conv[0][0], layers_conv[0][1], 0.5)
                ),
                steps,
                time_size,
            ),
            *[
                ConvWrapper(ConvBlock(c_i, c_o, 0.5))
                for c_i, c_o in layers_conv[1:]
            ],
            *[
                ConvWrapper(ConvBlock(c_i, c_o, 2.0))
                for c_i, c_o in layer_conv_tr
            ],
            ConvWrapper(ConvEndBlock(layer_conv_tr[-1][1], self.__channels)),
        )

    def forward(self, x_t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 5
        assert x_t.size(1) == self.__steps

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        t = th.arange(self.__steps, device=device).flip([0])

        eps = self.__eps((x_t, t))
        out: th.Tensor = (1.0 / th.sqrt(self.alphas)) * (
            x_t - eps * self.betas / th.sqrt(1.0 - self.alpha_cumprod)
        )

        return out

    def sample(self, x_t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        for t in range(1, self.__steps):
            z = (
                th.randn_like(x_t, device=device)
                if t > 1
                else th.zeros_like(x_t, device=device)
            )

            eps = self.__eps(
                (
                    x_t.unsqueeze(1),
                    th.tensor([self.__steps - t], device=device),
                )
            ).squeeze(1)

            x_t = (1.0 / th.sqrt(self.alphas[:, t])) * (
                x_t
                - eps
                * (1.0 - self.alphas[:, t])
                / th.sqrt(1.0 - self.alpha_cumprod[:, t])
            ) + self.betas[:, t] * z

        return x_t
