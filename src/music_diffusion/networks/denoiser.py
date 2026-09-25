from statistics import mean

import numpy as np
import torch as th
from tqdm import tqdm

from .diffusion import AbstractDiffuser
from .functions import select_time_scheduler
from .init import weights_init
from .unet import TimeUNet


class Denoiser(AbstractDiffuser):
    def __init__(
        self,
        steps: int,
        time_size: int,
        unet_channels: list[tuple[int, int]],
        unet_group_norm_nums: list[int],
    ) -> None:
        super().__init__(steps)

        self.__channels = unet_channels[0][0]

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
            unet_channels,
            unet_group_norm_nums,
            time_size,
            self._steps,
        )

        self.apply(weights_init)

    def forward(
        self, x_t: th.Tensor, t: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
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
        alphas_cum_prod: th.Tensor | None,
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
        alphas: th.Tensor | None = None,
        betas: th.Tensor | None = None,
        alphas_cum_prod: th.Tensor | None = None,
        alphas_cum_prod_prev: th.Tensor | None = None,
    ) -> th.Tensor:
        x_0_clipped = self.__x0_from_noise(x_t, eps_theta, t, alphas_cum_prod)

        mu: th.Tensor = self._mu_tiddle(
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
        betas: th.Tensor | None = None,
        betas_tiddle: th.Tensor | None = None,
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
    ) -> tuple[th.Tensor, th.Tensor]:
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
        betas_s = th.clamp_max(betas_s, 0.999)

        betas_tiddle_s = (
            betas_s
            * (1.0 - alphas_cum_prod_prev_s)
            / (1.0 - alphas_cum_prod_s)
        )
        betas_tiddle_s = th.clamp_min(betas_tiddle_s, betas_tiddle_s[1])

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
