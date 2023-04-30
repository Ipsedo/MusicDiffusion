from typing import Optional, Tuple

import torch as th
from torch import nn

from .functions import normal_cdf, process_factor


class Noiser(nn.Module):
    def __init__(self, steps: int, beta_1: float, beta_t: float) -> None:
        super().__init__()

        self.__steps = steps

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)

        alphas = 1 - betas
        alphas_cum_prod = th.cumprod(alphas, dim=0)
        sqrt_alphas_cum_prod = th.sqrt(alphas_cum_prod)
        sqrt_one_minus_alphas_cum_prod = th.sqrt(1 - alphas_cum_prod)

        alphas_cum_prod_prev = th.cat(
            [th.tensor([1.0]), alphas_cum_prod[:-1]], dim=0
        )

        betas_bar = (
            betas * (1.0 - alphas_cum_prod_prev) / (1.0 - alphas_cum_prod)
        )

        self.alphas: th.Tensor
        self.alphas_cum_prod: th.Tensor
        self.alphas_cum_prod_prev: th.Tensor
        self.sqrt_alphas_cum_prod: th.Tensor
        self.sqrt_one_minus_alphas_cum_prod: th.Tensor

        self.betas: th.Tensor
        self.betas_bar: th.Tensor

        self.register_buffer(
            "alphas",
            alphas,
        )
        self.register_buffer(
            "alphas_cum_prod",
            alphas_cum_prod,
        )
        self.register_buffer(
            "alphas_cum_prod_prev",
            alphas_cum_prod_prev,
        )
        self.register_buffer(
            "sqrt_alphas_cum_prod",
            sqrt_alphas_cum_prod,
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cum_prod",
            sqrt_one_minus_alphas_cum_prod,
        )

        # pylint: disable=duplicate-code
        self.register_buffer(
            "betas",
            betas,
        )
        self.register_buffer(
            "betas_bar",
            betas_bar,
        )
        # pylint: enable=duplicate-code

    def forward(
        self, x_0: th.Tensor, t: th.Tensor, eps: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x_0.size()) == 4
        assert len(t.size()) == 2
        assert x_0.size(0) == t.size(0)

        b, c, w, h = x_0.size()
        nb_steps = t.size(1)

        device = "cuda" if next(self.buffers()).is_cuda else "cpu"

        if eps is None:
            eps = th.randn(b, nb_steps, c, w, h, device=device)

        sqrt_alphas_cum_prod = process_factor(self.sqrt_alphas_cum_prod, t)
        sqrt_one_minus_alphas_cum_prod = process_factor(
            self.sqrt_one_minus_alphas_cum_prod, t
        )

        x_t = (
            sqrt_alphas_cum_prod * x_0.unsqueeze(1)
            + eps * sqrt_one_minus_alphas_cum_prod
        )

        return x_t, eps

    def __mu(self, x_t: th.Tensor, x_0: th.Tensor, t: th.Tensor) -> th.Tensor:
        alphas_cum_prod_prev = process_factor(self.alphas_cum_prod_prev, t)
        alphas_cum_prod = process_factor(self.alphas_cum_prod, t)
        alphas = process_factor(self.alphas, t)
        betas = process_factor(self.betas, t)

        mu: th.Tensor = x_0.unsqueeze(1) * th.sqrt(
            alphas_cum_prod_prev
        ) * betas / (1.0 - alphas_cum_prod) + x_t * th.sqrt(alphas) * (
            1 - alphas_cum_prod_prev
        ) / (
            1.0 - alphas_cum_prod
        )
        return mu

    def posterior(
        self,
        x_t_prev: th.Tensor,
        x_t: th.Tensor,
        x_0: th.Tensor,
        t: th.Tensor,
        epsilon: float = 1e-8,
    ) -> th.Tensor:
        assert len(x_0.size()) == 4
        assert len(t.size()) == 2
        assert x_0.size(0) == t.size(0)

        betas_bar = process_factor(self.betas_bar, t)
        betas_bar = betas_bar.repeat(
            1, 1, x_t.size(2), x_t.size(3), x_t.size(4)
        )

        posterior: th.Tensor = normal_cdf(
            x_t_prev, self.__mu(x_t, x_0, t), betas_bar + epsilon
        )

        return posterior
