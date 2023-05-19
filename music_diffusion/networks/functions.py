# -*- coding: utf-8 -*-
from math import sqrt

import torch as th
from torch.distributions import Normal


def normal_log_prob(
    x: th.Tensor, mu: th.Tensor, sigma: th.Tensor
) -> th.Tensor:
    dist = Normal(mu, sigma)
    log_prob: th.Tensor = dist.log_prob(x)
    return log_prob


def normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    return th.exp(-0.5 * th.pow((x - mu) / sigma, 2.0)) / th.sqrt(
        2 * th.pi * sigma**2
    )


def normal_cdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    dist = Normal(mu, sigma)
    cdf: th.Tensor = dist.cdf(x)
    return cdf


def log_normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    log_density = th.exp(
        -th.pow(th.log(x) - mu, 2.0) / (2.0 * th.pow(sigma, 2.0))
    ) / (x * sigma * sqrt(2 * th.pi))

    return log_density.flatten(2, -1)


def select_time_scheduler(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))


def hellinger(p: th.Tensor, q: th.Tensor, epsilon: float = 1e-8) -> th.Tensor:
    return 2 * th.pow(th.sqrt(p + epsilon) - th.sqrt(q + epsilon), 2)


def kl_div(p: th.Tensor, q: th.Tensor, epsilon: float = 1e-8) -> th.Tensor:
    return p * (th.log(p + epsilon) - th.log(q + epsilon))
