# -*- coding: utf-8 -*-
from abc import ABC
from statistics import mean
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm

from .functions import select_time_scheduler
from .init import weights_init
from .unet import TimeUNet


class Diffuser(ABC, nn.Module):
    def __init__(self, steps: int, beta_1: float, beta_t: float):
        super().__init__()

        self._steps = steps

        # unused
        self._beta_1 = beta_1
        self._beta_t = beta_t

        # betas = th.linspace(self._beta_1, self._beta_t, steps=self._steps)

        # time schedulers improved
        # 8e-4
        s = 1e-8

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
        # 1e-4 and 0.999
        betas = th.clamp(betas, self._beta_1, 1 - self._beta_1)

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
        self._betas_tiddle_limit = 1e-20
        betas_tiddle = th.clamp_min(betas_tiddle, self._betas_tiddle_limit)

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


##########
# Noising
##########


class Noiser(Diffuser):
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

        sqrt_alphas_cum_prod = select_time_scheduler(
            self._sqrt_alphas_cum_prod, t
        )
        sqrt_one_minus_alphas_cum_prod = select_time_scheduler(
            self._sqrt_one_minus_alphas_cum_prod, t
        )

        x_t = (
            sqrt_alphas_cum_prod * x_0.unsqueeze(1)
            + eps * sqrt_one_minus_alphas_cum_prod
        )

        return x_t, eps

    def __mu(self, x_t: th.Tensor, x_0: th.Tensor, t: th.Tensor) -> th.Tensor:
        return super()._mu_tiddle(x_t, x_0.unsqueeze(1), t)

    def __var(self, t: th.Tensor) -> th.Tensor:

        betas: th.Tensor = select_time_scheduler(self._betas, t)

        return betas

    def posterior(
        self,
        x_t: th.Tensor,
        x_0: th.Tensor,
        t: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 5
        assert len(x_0.size()) == 4
        assert len(t.size()) == 2

        return self.__mu(x_t, x_0, t), self.__var(t)


############
# Denoising
############


class Denoiser(Diffuser):
    def __init__(
        self,
        channels: int,
        steps: int,
        time_size: int,
        beta_1: float,
        beta_t: float,
        unet_channels: List[Tuple[int, int]],
        norm_groups: int,
    ) -> None:
        super().__init__(steps, beta_1, beta_t)

        self.__channels = channels

        self._sqrt_alpha: th.Tensor
        self._sqrt_betas: th.Tensor

        self.register_buffer(
            "_sqrt_alpha",
            th.sqrt(self._alphas),
        )

        self.register_buffer(
            "_sqrt_betas",
            th.sqrt(self._betas),
        )

        self.__unet = TimeUNet(
            channels,
            channels,
            unet_channels,
            time_size,
            norm_groups,
            self._steps,
        )

        self.apply(weights_init)

    def forward(
        self, x_t: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 5
        assert len(t.size()) == 2
        assert x_t.size(0) == t.size(0)
        assert x_t.size(1) == t.size(1)

        eps_theta, v_theta = self.__unet(x_t, t)

        return eps_theta, v_theta

    def __x0_from_noise(
        self,
        x_t: th.Tensor,
        eps: th.Tensor,
        t: th.Tensor,
        alphas_cum_prod: Optional[th.Tensor],
    ) -> th.Tensor:
        alphas_cum_prod = (
            select_time_scheduler(self._alphas_cum_prod, t)
            if alphas_cum_prod is None
            else alphas_cum_prod
        )
        x_0: th.Tensor = (x_t - eps * th.sqrt(1 - alphas_cum_prod)) / th.sqrt(
            alphas_cum_prod
        )
        return th.clip(x_0, -1.0, 1.0)

    def __mu_clipped(
        self,
        x_t: th.Tensor,
        eps_theta: th.Tensor,
        t: th.Tensor,
        alphas: Optional[th.Tensor] = None,
        betas: Optional[th.Tensor] = None,
        alphas_cum_prod: Optional[th.Tensor] = None,
        alphas_cum_prod_prev: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        x_0_clipped = self.__x0_from_noise(x_t, eps_theta, t, alphas_cum_prod)

        mu = self._mu_tiddle(
            x_t,
            x_0_clipped,
            t,
            alphas,
            betas,
            alphas_cum_prod,
            alphas_cum_prod_prev,
        )

        return mu

    def __mu(
        self, x_t: th.Tensor, eps_theta: th.Tensor, t: th.Tensor
    ) -> th.Tensor:

        mu: th.Tensor = (
            x_t
            - eps_theta
            * select_time_scheduler(self._betas, t)
            / select_time_scheduler(self._sqrt_one_minus_alphas_cum_prod, t)
        ) / select_time_scheduler(self._sqrt_alpha, t)
        return mu

    def __var(
        self,
        v: th.Tensor,
        t: th.Tensor,
        betas: Optional[th.Tensor] = None,
        betas_tiddle: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        betas = (
            select_time_scheduler(self._betas, t) if betas is None else betas
        )
        betas_tiddle = (
            select_time_scheduler(self._betas_tiddle, t)
            if betas_tiddle is None
            else betas_tiddle
        )

        return th.exp(v * th.log(betas) + (1.0 - v) * th.log(betas_tiddle))

    def prior(
        self,
        x_t: th.Tensor,
        t: th.Tensor,
        eps_theta: th.Tensor,
        v_theta: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 5
        assert len(t.size()) == 2
        assert len(eps_theta.size()) == 5

        return self.__mu(x_t, eps_theta, t), self.__var(v_theta, t)

    @th.no_grad()
    def sample(self, x_t: th.Tensor, verbose: bool = False) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        times = list(reversed(range(self._steps)))
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for t in tqdm_bar:
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            t_tensor = th.tensor([[t]], device=device)

            eps, v = self.__unet(
                x_t.unsqueeze(1),
                t_tensor.repeat(x_t.size(0), 1),
            )

            # original sampling method
            # see : https://github.com/hojonathanho/diffusion/issues/5
            # see : https://github.com/openai/improved-diffusion/issues/64
            mu = self.__mu_clipped(x_t.unsqueeze(1), eps, t_tensor).squeeze(1)
            sigma = self.__var(v, t_tensor).sqrt().squeeze(1)

            x_t = mu + sigma * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    @th.no_grad()
    def fast_sample(
        self, x_t: th.Tensor, n_steps: int, verbose: bool = False
    ) -> th.Tensor:
        assert len(x_t.size()) == 4
        assert x_t.size(1) == self.__channels

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        steps = th.linspace(
            0, self._steps - 1, steps=n_steps, dtype=th.long, device=device
        )

        alphas_cum_prod_s = self._alphas_cum_prod[steps]
        # alphas_cum_prod_prev_s = self._alphas_cum_prod_prev[steps]
        alphas_cum_prod_prev_s = th.cat(
            [th.tensor([1], device=device), alphas_cum_prod_s[:-1]], dim=0
        )

        betas_s = 1.0 - alphas_cum_prod_s / alphas_cum_prod_prev_s
        betas_s = th.clamp(betas_s, self._beta_1, 1 - self._beta_1)

        betas_tiddle_s = (
            betas_s
            * (1.0 - alphas_cum_prod_prev_s)
            / (1.0 - alphas_cum_prod_s)
        )
        betas_tiddle_s = th.clamp_min(betas_tiddle_s, self._betas_tiddle_limit)

        alphas_s = 1.0 - betas_s

        times = steps.flip(0).cpu().numpy().tolist()
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for s_t, t in enumerate(tqdm_bar):
            s_t = len(times) - s_t - 1

            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            eps, v = self.__unet(
                x_t.unsqueeze(1),
                th.tensor([[t]], device=device).repeat(x_t.size(0), 1),
            )

            mu = self.__mu_clipped(
                x_t.unsqueeze(1),
                eps,
                t,
                alphas_s[s_t, None, None],
                betas_s[s_t, None, None],
                alphas_cum_prod_s[s_t, None, None],
                alphas_cum_prod_prev_s[s_t, None, None],
            )
            mu = mu.squeeze(1)

            var = self.__var(
                v, t, betas_s[s_t, None, None], betas_tiddle_s[s_t, None, None]
            )
            var = var.squeeze(1)

            x_t = mu + var.sqrt() * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    def loss_factor(self, t: th.Tensor) -> th.Tensor:
        assert len(t.size()) == 2

        alphas = select_time_scheduler(self._alphas, t)
        betas = select_time_scheduler(self._betas, t)
        alphas_cum_prod = select_time_scheduler(self._alphas_cum_prod, t)

        # sig^2 = beta
        scale: th.Tensor = betas / (2.0 * alphas * (1.0 - alphas_cum_prod))

        return scale[:, :, None, None, None]

    def count_parameters(self) -> int:
        return int(
            sum(
                np.prod(p.size()) for p in self.parameters() if p.requires_grad
            )
        )

    def grad_norm(self) -> float:
        return float(
            mean(
                p.grad.norm().item()
                for p in self.parameters()
                if p.grad is not None
            )
        )
