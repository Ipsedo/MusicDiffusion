from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm

from .init import weights_init
from .unet import TimeUNet


class Denoiser(nn.Module):
    def __init__(
        self,
        channels: int,
        steps: int,
        time_size: int,
        beta_1: float,
        beta_t: float,
        unet_channels: List[Tuple[int, int]],
        use_attentions: List[bool],
        attention_heads: int,
    ) -> None:
        super().__init__()

        self.__steps = steps

        self.__channels = channels

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)
        alphas = 1.0 - betas
        alpha_cum_prod = th.cumprod(alphas, dim=0)

        self.alphas: th.Tensor
        self.sqrt_alpha: th.Tensor
        self.alpha_cum_prod: th.Tensor
        self.sqrt_one_minus_alpha_cum_prod: th.Tensor
        self.betas: th.Tensor
        self.sqrt_betas: th.Tensor

        self.register_buffer(
            "alphas",
            alphas,
        )
        self.register_buffer(
            "sqrt_alpha",
            th.sqrt(alphas),
        )
        self.register_buffer(
            "alpha_cum_prod",
            alpha_cum_prod,
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_cum_prod",
            th.sqrt(1.0 - self.alpha_cum_prod),
        )
        self.register_buffer(
            "betas",
            betas,
        )
        self.register_buffer(
            "sqrt_betas",
            th.sqrt(self.betas),
        )

        self.__eps = TimeUNet(
            channels,
            channels,
            unet_channels,
            use_attentions,
            attention_heads,
            time_size,
            self.__steps,
        )

        self.apply(weights_init)

    def forward(self, x_t: th.Tensor, t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 5
        assert len(t.size()) == 2
        assert x_t.size(0) == t.size(0)
        assert x_t.size(1) == t.size(1)

        eps_theta: th.Tensor = self.__eps(x_t, t)

        return eps_theta

    def sample(self, x_t: th.Tensor, verbose: bool = False) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        times = list(reversed(range(self.__steps)))
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for t in tqdm_bar:
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            eps = self.__eps(
                x_t.unsqueeze(1),
                th.tensor([[t]], device=device).repeat(x_t.size(0), 1),
            ).squeeze(1)

            x_t = (1.0 / self.sqrt_alpha[t]) * (
                x_t
                - eps
                * (1.0 - self.alphas[t])
                / self.sqrt_one_minus_alpha_cum_prod[t]
            ) + self.sqrt_betas[t] * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    def loss_factor(self, t: th.Tensor) -> th.Tensor:
        assert len(t.size()) == 2
        batch_size, steps = t.size()

        t = t.flatten(0, 1)

        scale: th.Tensor = self.betas[t] / (
            2.0 * self.alphas[t] * (1.0 - self.alpha_cum_prod[t])
        )

        scale = th.unflatten(scale, 0, (batch_size, steps))

        return scale[:, :, None, None, None]

    def count_parameters(self) -> int:
        return int(
            np.sum(
                [
                    np.prod(p.size())
                    for p in self.parameters()
                    if p.requires_grad
                ]
            )
        )
