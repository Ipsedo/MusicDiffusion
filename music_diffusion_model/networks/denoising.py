import torch as th
import torch.nn as nn

from .convolutions import AbstractConv, ConvBlock, ConvEndBlock, StrideConv
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

        new_w, new_h = (
            int(w * self.__conv.scale_factor),
            int(h * self.__conv.scale_factor),
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
    ) -> None:
        super().__init__()

        self.__steps = steps

        self.__channels = channels

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)
        alphas = 1.0 - betas
        alpha_cumprod = th.cumprod(alphas, dim=0)

        self.alphas: th.Tensor
        self.alpha_cumprod: th.Tensor
        self.betas: th.Tensor

        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("betas", betas)

        encoder_layers = [
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
        ]

        decoder_layers = [
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16),
        ]

        self.__eps = nn.Sequential(
            TimeWrapper(
                ConvWrapper(
                    ConvBlock(
                        self.__channels + time_size,
                        encoder_layers[0][0],
                        scale_factor=1.0,
                    )
                ),
                steps,
                time_size,
            ),
            *[
                ConvWrapper(
                    StrideConv(
                        c_i,
                        c_o,
                        scale="down",
                    )
                )
                for c_i, c_o in encoder_layers
            ],
            *[
                ConvWrapper(
                    StrideConv(
                        c_i,
                        c_o,
                        scale="up",
                    )
                )
                for c_i, c_o in decoder_layers
            ],
            ConvWrapper(
                ConvEndBlock(
                    decoder_layers[-1][1],
                    self.__channels,
                )
            ),
        )

    def forward(self, x_0_to_t: th.Tensor, t: th.Tensor) -> th.Tensor:
        assert len(x_0_to_t.size()) == 5
        assert x_0_to_t.size(0) == t.size(0)
        assert x_0_to_t.size(1) == t.size(1)

        eps_theta: th.Tensor = self.__eps((x_0_to_t, t))

        return eps_theta

    def sample(self, x_t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        for t in reversed(range(self.__steps)):
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            eps = self.__eps(
                (
                    x_t.unsqueeze(1),
                    th.tensor([[t]], device=device).repeat(x_t.size(0), 1),
                )
            ).squeeze(1)

            x_t = (1.0 / th.sqrt(self.alphas[None, t])) * (
                x_t
                - eps
                * (1.0 - self.alphas[None, t])
                / th.sqrt(1.0 - self.alpha_cumprod[None, t])
            ) + th.sqrt(self.betas[None, t]) * z

        return x_t

    def loss_scale(self, t: th.Tensor) -> th.Tensor:
        assert len(t.size()) == 2
        b, s = t.size()

        t = t.flatten()

        betas = th.index_select(self.betas, dim=0, index=t)
        alphas = th.index_select(self.alphas, dim=0, index=t)
        alpha_cumprod = th.index_select(self.alpha_cumprod, dim=0, index=t)

        scale: th.Tensor = betas / (2.0 * alphas * (1.0 - alpha_cumprod))
        return scale.view(b, s)[:, :, None, None, None]
