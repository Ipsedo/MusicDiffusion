# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
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

        """s = 8e-4

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
        betas[betas > 0.999] = 0.999
        betas[betas < 1e-4] = 1e-4
        alphas = 1 - betas"""

        betas = th.linspace(beta_1, beta_t, steps=self._steps)

        alphas = 1 - betas

        alphas_cum_prod = th.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = th.cat(
            [th.tensor([1]), alphas_cum_prod[:-1]], dim=0
        )

        # alphas_cum_prod = alphas_cum_prod[1:]
        # alphas = alphas[1:]
        # betas = betas[1:]

        sqrt_alphas_cum_prod = th.sqrt(alphas_cum_prod)
        sqrt_one_minus_alphas_cum_prod = th.sqrt(1 - alphas_cum_prod)

        self._betas_tiddle_limit = 1e-10
        betas_tiddle = (
            betas * (1.0 - alphas_cum_prod_prev) / (1.0 - alphas_cum_prod)
        )
        betas_tiddle[
            betas_tiddle < self._betas_tiddle_limit
        ] = self._betas_tiddle_limit

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

    @abstractmethod
    def _mu(self, *args: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _var(self, *args: th.Tensor) -> th.Tensor:
        pass


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

    def _mu(self, *args: th.Tensor) -> th.Tensor:
        x_t, x_0, t = args

        alphas_cum_prod_prev = select_time_scheduler(
            self._alphas_cum_prod_prev, t
        )

        betas = select_time_scheduler(self._betas, t)

        alphas_cum_prod = select_time_scheduler(self._alphas_cum_prod, t)

        alphas = select_time_scheduler(self._alphas, t)

        mu: th.Tensor = x_0.unsqueeze(1) * th.sqrt(
            alphas_cum_prod_prev
        ) * betas / (1 - alphas_cum_prod) + x_t * th.sqrt(alphas) * (
            1 - alphas_cum_prod_prev
        ) / (
            1 - alphas_cum_prod
        )

        return mu

    # pylint: disable=duplicate-code
    def _var(self, *args: th.Tensor) -> th.Tensor:
        (t,) = args

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

        return self._mu(x_t, x_0, t), self._var(t)


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
            self._steps,
        )

        self.apply(weights_init)

    def forward(self, x_t: th.Tensor, t: th.Tensor) -> th.Tensor:
        assert len(x_t.size()) == 5
        assert len(t.size()) == 2
        assert x_t.size(0) == t.size(0)
        assert x_t.size(1) == t.size(1)

        eps_theta: th.Tensor = self.__unet(x_t, t)

        return eps_theta

    @staticmethod
    def __mu_generic(
        x_t: th.Tensor,
        eps_theta: th.Tensor,
        sqrt_alpha: th.Tensor,
        betas: th.Tensor,
        alphas_cum_prod: th.Tensor,
        alphas_cum_prod_prev: th.Tensor,
    ) -> th.Tensor:
        x_t_next_clipped = th.clip(
            x_t / th.sqrt(alphas_cum_prod)
            - eps_theta * th.sqrt((1 - alphas_cum_prod) / alphas_cum_prod),
            -1,
            1,
        )

        x_t_next_arg = (
            x_t
            * (1 - alphas_cum_prod_prev)
            * sqrt_alpha
            / (1 - alphas_cum_prod)
        )

        mu: th.Tensor = (
            th.sqrt(alphas_cum_prod_prev)
            * betas
            * x_t_next_clipped
            / (1 - alphas_cum_prod)
            + x_t_next_arg
        )

        return mu

    @staticmethod
    def __var_generic(betas: th.Tensor) -> th.Tensor:
        return betas

    def _mu(self, *args: th.Tensor) -> th.Tensor:
        x_t, eps_theta, t = args

        sqrt_alpha = select_time_scheduler(self._sqrt_alpha, t)
        betas = select_time_scheduler(self._betas, t)

        alphas_cum_prod = select_time_scheduler(self._alphas_cum_prod, t)
        alphas_cum_prod_prev = select_time_scheduler(
            self._alphas_cum_prod_prev, t
        )

        return Denoiser.__mu_generic(
            x_t,
            eps_theta,
            sqrt_alpha,
            betas,
            alphas_cum_prod,
            alphas_cum_prod_prev,
        )

    def _var(self, *args: th.Tensor) -> th.Tensor:
        (t,) = args

        betas: th.Tensor = select_time_scheduler(self._betas, t)

        return Denoiser.__var_generic(betas)

    def prior(
        self,
        x_t: th.Tensor,
        t: th.Tensor,
        eps_theta: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 5
        assert len(t.size()) == 2
        assert len(eps_theta.size()) == 5

        return self._mu(x_t, eps_theta, t), self._var(t)

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

            eps = self.__unet(
                x_t.unsqueeze(1),
                t_tensor.repeat(x_t.size(0), 1),
            )

            # original sampling method
            # see : https://github.com/hojonathanho/diffusion/issues/5
            mu = self._mu(x_t.unsqueeze(1), eps, t_tensor).squeeze(1)
            sigma = self._var(t_tensor).sqrt().squeeze(1)

            x_t = mu + sigma * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    def fast_sample(
        self, x_t: th.Tensor, n_steps: int, verbose: bool = False
    ) -> th.Tensor:

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        steps = th.floor(
            th.linspace(0, self._steps - 1, steps=n_steps, device=device)
        ).to(th.int)

        alphas_cum_prod_s = self._alphas_cum_prod[steps]
        alphas_cum_prod_prev_s = self._alphas_cum_prod_prev[steps]

        betas_s = 1 - alphas_cum_prod_s / alphas_cum_prod_prev_s

        alphas_s = 1 - betas_s

        times = steps.cpu().numpy().tolist()
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for s_t, t in enumerate(tqdm_bar):
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            eps = self.__unet(
                x_t.unsqueeze(1),
                th.tensor([[t]], device=device).repeat(x_t.size(0), 1),
            )

            mu = Denoiser.__mu_generic(
                x_t.unsqueeze(1),
                eps,
                th.sqrt(alphas_s[s_t, None, None]),
                betas_s[s_t, None, None],
                alphas_cum_prod_s[s_t, None, None],
                alphas_cum_prod_prev_s[s_t, None, None],
            )
            mu = mu.squeeze(1)

            var = Denoiser.__var_generic(betas_s[s_t, None, None])
            var = var.squeeze(1)

            x_t = mu + var.sqrt() * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    def loss_factor(self, t: th.Tensor) -> th.Tensor:
        assert len(t.size()) == 2
        batch_size, steps = t.size()

        t = t.flatten(0, 1)

        scale: th.Tensor = self._betas[t] / (
            2.0 * self._alphas[t] * (1.0 - self._alphas_cum_prod[t])
        )

        scale = th.unflatten(scale, 0, (batch_size, steps))

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
