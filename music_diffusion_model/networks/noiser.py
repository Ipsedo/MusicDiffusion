import torch as th
import torch.nn as nn


class Noiser(nn.Module):
    def __init__(self, steps: int, beta_1: float, beta_t: float) -> None:
        super().__init__()

        self.__steps = steps

        betas = th.linspace(beta_1, beta_t, steps=self.__steps)

        alphas = 1 - betas
        alphas_cum_prod = th.cumprod(alphas, dim=0)
        self.__sqrt_alphas_cum_prod = th.sqrt(alphas_cum_prod)[
            None, :, None, None, None
        ]
        self.__sqrt_minus_one_alphas_cum_prod = th.sqrt(1 - alphas_cum_prod)[
            None, :, None, None, None
        ]

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4

        noise = th.rand_like(x.unsqueeze(1).repeat(1, self.__steps, 1, 1, 1))

        x_t = (
            self.__sqrt_alphas_cum_prod * x.unsqueeze(1)
            + self.__sqrt_minus_one_alphas_cum_prod * noise
        )

        return x_t
