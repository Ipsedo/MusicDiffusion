from typing import List, Optional, Tuple

import torch as th
import torch.nn as nn


class Noiser(nn.Module):
    def __init__(self, steps: int, beta_1: float, beta_t: float) -> None:
        super().__init__()

        self.__steps = steps

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)

        alphas = 1 - betas
        alphas_cum_prod = th.cumprod(alphas, dim=0)
        sqrt_alphas_cum_prod = th.sqrt(alphas_cum_prod)[
            None, :, None, None, None
        ]
        sqrt_minus_one_alphas_cum_prod = th.sqrt(1 - alphas_cum_prod)[
            None, :, None, None, None
        ]

        self.sqrt_alphas_cum_prod: th.Tensor
        self.sqrt_minus_one_alphas_cum_prod: th.Tensor

        self.register_buffer("sqrt_alphas_cum_prod", sqrt_alphas_cum_prod)
        self.register_buffer(
            "sqrt_minus_one_alphas_cum_prod", sqrt_minus_one_alphas_cum_prod
        )

    def forward(
        self, x: th.Tensor, t: Optional[List[int]] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x.size()) == 4
        b, c, w, h = x.size()

        device = "cuda" if next(self.buffers()).is_cuda else "cpu"

        eps = th.randn(b, 1, c, w, h, device=device)
        eps = eps.repeat(1, self.__steps if t is None else len(t), 1, 1, 1)

        sqrt_alphas_cum_prod, sqrt_minus_one_alphas_cum_prod = (
            (self.sqrt_alphas_cum_prod, self.sqrt_minus_one_alphas_cum_prod)
            if t is None
            else (
                th.index_select(
                    self.sqrt_alphas_cum_prod, dim=1, index=th.tensor(t)
                ),
                th.index_select(
                    self.sqrt_minus_one_alphas_cum_prod,
                    dim=1,
                    index=th.tensor(t),
                ),
            )
        )

        x_t = (
            sqrt_alphas_cum_prod * x.unsqueeze(1)
            + eps * sqrt_minus_one_alphas_cum_prod
        )

        return x_t, eps
