# -*- coding: utf-8 -*-
from math import log, sqrt
from typing import Tuple

import torch as th
from torch.distributions import Normal

from music_diffusion.data import BIN_SIZE


def select_time_scheduler(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))


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


def hellinger(p: th.Tensor, q: th.Tensor, epsilon: float = 1e-8) -> th.Tensor:
    return 2 * th.pow(th.sqrt(p + epsilon) - th.sqrt(q + epsilon), 2)


def kl_div(p: th.Tensor, q: th.Tensor, epsilon: float = 1e-8) -> th.Tensor:
    return p * (th.log(p + epsilon) - th.log(q + epsilon))


def log_kl_div(log_p: th.Tensor, log_q: th.Tensor) -> th.Tensor:
    return log_p.exp() * (log_p - log_q)


def normal_kl_div(
    mu_1: th.Tensor,
    var_1: th.Tensor,
    mu_2: th.Tensor,
    var_2: th.Tensor,
    epsilon: float = 1e-12,
) -> th.Tensor:
    return (
        th.log(var_2 + epsilon) / 2.0
        - th.log(var_1 + epsilon) / 2.0
        + (var_1 + th.pow(mu_1 - mu_2, 2.0)) / (2 * var_2 + epsilon)
        - 0.5
        # .sum(dim=[2, 3, 4])
        # .clamp_max(clip_max)
        # .div(div_factor)
    )


def mse(p: th.Tensor, q: th.Tensor) -> th.Tensor:
    return th.pow(p - q, 2.0)  # .mean(dim=[2, 3, 4])


def normal_bhattacharyya(
    mu_1: th.Tensor,
    var_1: th.Tensor,
    mu_2: th.Tensor,
    var_2: th.Tensor,
) -> th.Tensor:
    sig_1 = th.sqrt(var_1)
    sig_2 = th.sqrt(var_2)
    sigma = (sig_1 + sig_2) / 2.0

    return th.sum(
        th.pow(mu_1 - mu_2, 2.0) / sigma / 8.0
        + th.log(sigma / th.sqrt(sig_1 * sig_2)),
        dim=[2, 3, 4],
    )


def normal_wasserstein(
    mu_1: th.Tensor,
    var_1: th.Tensor,
    mu_2: th.Tensor,
    var_2: th.Tensor,
) -> th.Tensor:

    return th.sum(
        th.pow(mu_1 - mu_2, 2.0)
        + th.pow(th.sqrt(var_1) - th.sqrt(var_2), 2.0),
        dim=[2, 3, 4],
    )


def negative_log_likelihood(
    x: th.Tensor,
    mu: th.Tensor,
    var: th.Tensor,
    epsilon: float = 1e-10,
    clip_max: float = 128.0,
) -> th.Tensor:
    return (
        (
            0.5
            * (th.log(2 * th.pi * var) + th.pow(x - mu, 2.0) / (var + epsilon))
        )
        .clamp(0, clip_max)
        .mean(dim=[2, 3, 4])
    )


def discretized_nll(
    x: th.Tensor,
    mu: th.Tensor,
    var: th.Tensor,
    precision: float = BIN_SIZE,
    cut_off: Tuple[float, float] = (-0.999, 0.999),
    epsilon: float = 1e-20,
) -> th.Tensor:
    div_factor = -log(epsilon)

    min_cut_off, max_cut_off = cut_off
    sigma = th.sqrt(var)

    cdf_plus = th.where(
        th.lt(x, max_cut_off),
        normal_cdf(x + precision, mu, sigma),
        th.tensor(1.0, device=x.device),
    )

    cdf_min = th.where(
        th.gt(x, min_cut_off),
        normal_cdf(x - precision, mu, sigma),
        th.tensor(0.0, device=x.device),
    )

    return (
        th.log(th.clamp_min(cdf_plus - cdf_min, epsilon))
        .mul(-1)
        .mean(dim=[2, 3, 4])
        .div(div_factor)
    )
