from abc import ABC
from typing import Optional

import torch as th
from torch import nn

from music_diffusion.data import BIN_SIZE

from .functions import select_time_scheduler


class AbstractDiffuser(ABC, nn.Module):
    def __init__(self, steps: int):
        super().__init__()

        self._steps = steps

        # betas = th.linspace(self._beta_1, self._beta_t, steps=self._steps)

        # time schedulers improved
        # 16-bit audio : 1/(2^16/2) ?
        s = BIN_SIZE

        linear_space: th.Tensor = th.linspace(0.0, 1.0, steps=self._steps + 1)
        # exponent: th.Tensor = linear_space * 3. * th.pi - 1.5 * th.pi
        # exponent = -exponent
        # f_values = 1 - 1. / (1. + th.exp(exponent))
        f_values = th.pow(
            th.cos(0.5 * th.pi * (linear_space + s) / (1 + s)), 2.0
        )

        alphas_cum_prod = f_values[1:] / f_values[0]
        alphas_cum_prod_prev = f_values[:-1] / f_values[0]

        betas = 1 - alphas_cum_prod / alphas_cum_prod_prev
        betas = th.clamp_max(betas, 0.999)

        alphas = 1 - betas

        alphas_cum_prod = th.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = th.cat(
            [th.tensor([1]), alphas_cum_prod[:-1]], dim=0
        )

        sqrt_alphas_cum_prod = th.sqrt(alphas_cum_prod)
        sqrt_one_minus_alphas_cum_prod = th.sqrt(1 - alphas_cum_prod)

        betas_tiddle = (
            betas * (1.0 - alphas_cum_prod_prev) / (1.0 - alphas_cum_prod)
        )
        betas_tiddle = th.clamp_min(betas_tiddle, betas_tiddle[1])

        # attributes definition

        self._betas: th.Tensor

        self._alphas: th.Tensor
        self._alphas_cum_prod: th.Tensor

        self._sqrt_alphas_cum_prod: th.Tensor
        self._sqrt_one_minus_alphas_cum_prod: th.Tensor

        self._alphas_cum_prod_prev: th.Tensor

        self._betas_tiddle: th.Tensor

        # register buffers / time schedule

        self.register_buffer("_betas", betas)

        self.register_buffer("_alphas", alphas)
        self.register_buffer("_alphas_cum_prod", alphas_cum_prod)

        self.register_buffer("_sqrt_alphas_cum_prod", sqrt_alphas_cum_prod)
        self.register_buffer(
            "_sqrt_one_minus_alphas_cum_prod", sqrt_one_minus_alphas_cum_prod
        )

        self.register_buffer("_alphas_cum_prod_prev", alphas_cum_prod_prev)

        self.register_buffer("_betas_tiddle", betas_tiddle)

    def _mu_tiddle(
        self,
        x_t: th.Tensor,
        x_0: th.Tensor,
        t: th.Tensor,
        alphas: Optional[th.Tensor] = None,
        betas: Optional[th.Tensor] = None,
        alphas_cum_prod: Optional[th.Tensor] = None,
        alphas_cum_prod_prev: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        alphas_cum_prod_prev = (
            select_time_scheduler(self._alphas_cum_prod_prev, t)
            if alphas_cum_prod_prev is None
            else alphas_cum_prod_prev
        )

        betas = (
            select_time_scheduler(self._betas, t) if betas is None else betas
        )

        alphas_cum_prod = (
            select_time_scheduler(self._alphas_cum_prod, t)
            if alphas_cum_prod is None
            else alphas_cum_prod
        )

        alphas = (
            select_time_scheduler(self._alphas, t)
            if alphas is None
            else alphas
        )

        mu: th.Tensor = x_0 * th.sqrt(alphas_cum_prod_prev) * betas / (
            1 - alphas_cum_prod
        ) + x_t * th.sqrt(alphas) * (1 - alphas_cum_prod_prev) / (
            1 - alphas_cum_prod
        )

        return mu
