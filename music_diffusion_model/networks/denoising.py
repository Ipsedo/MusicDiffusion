from typing import List, Tuple

import torch as th
import torch.nn as nn
from tqdm import tqdm

from .time import TimeWrapper
from .unet import UNet


class Denoiser(nn.Module):
    def __init__(
        self,
        channels: int,
        steps: int,
        time_size: int,
        beta_1: float,
        beta_t: float,
        unet_channels: List[Tuple[int, int]],
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

        self.__eps = TimeWrapper(
            UNet(
                channels + time_size,
                channels,
                unet_channels,
            ),
            self.__steps,
            time_size,
        )

    def forward(self, x_0_to_t: th.Tensor, t: th.Tensor) -> th.Tensor:
        assert len(x_0_to_t.size()) == 5
        assert x_0_to_t.size(0) == t.size(0)
        assert x_0_to_t.size(1) == t.size(1)

        eps_theta: th.Tensor = self.__eps(x_0_to_t, t)

        return eps_theta

    def sample(self, x_t: th.Tensor, verbose: bool = False) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        times = reversed(range(self.__steps))
        if verbose:
            times = tqdm(times, leave=False)

        for t in times:
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            eps = self.__eps(
                x_t.unsqueeze(1),
                th.tensor([[t]], device=device).repeat(x_t.size(0), 1),
            ).squeeze(1)

            x_t = (1.0 / th.sqrt(self.alphas[t])) * (
                x_t
                - eps
                * (1.0 - self.alphas[t])
                / th.sqrt(1.0 - self.alpha_cumprod[t])
            ) + th.sqrt(self.betas[t]) * z

        return x_t
