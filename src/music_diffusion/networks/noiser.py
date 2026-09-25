import torch as th

from .diffusion import AbstractDiffuser
from .functions import select_time_scheduler


class Noiser(AbstractDiffuser):
    def forward(
        self, x_0: th.Tensor, t: th.Tensor, eps: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor]:
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
        mu: th.Tensor = self._mu_tiddle(x_t, x_0.unsqueeze(1), t)
        return mu

    def __var(self, t: th.Tensor) -> th.Tensor:

        betas: th.Tensor = select_time_scheduler(self._betas_tiddle, t)

        return betas

    def posterior(
        self,
        x_t: th.Tensor,
        x_0: th.Tensor,
        t: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 5
        assert len(x_0.size()) == 4
        assert len(t.size()) == 2

        return self.__mu(x_t, x_0, t), self.__var(t)
