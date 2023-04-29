from math import sqrt

import torch as th


def smart_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    value_range = max_value - min_value

    x = (x - min_value) / value_range % 1.0
    x[x < 0.0] += 1.0

    x = x * value_range + min_value

    return x


def bound_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    return th.clip(x, min_value, max_value)


def normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    """density = th.exp(-th.pow(x - mu, 2.0) / (2.0 * th.pow(sigma, 2.0))) \
    / th.sqrt(
        2.0 * th.pi * th.pow(sigma, 2.0)
    )
    normalization = th.sum(density, dim=(2, 3, 4), keepdim=True)
    probabilities = density / normalization
    return th.prod(probabilities.flatten(2, -1), dim=-1)"""
    k = x.size(2)

    exponent = -0.5 * ((x - mu) ** 2 / (sigma**2 + 1e-8)).sum(2)
    normalization_factor = 1 / (sqrt(2 * th.pi) ** k * (sigma + 1e-8)).prod(2)
    density: th.Tensor = normalization_factor * th.exp(exponent)
    return density.mean(dim=[-2, -1])


def log_normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    log_density = th.exp(
        -th.pow(th.log(x) - mu, 2.0) / (2.0 * th.pow(sigma, 2.0))
    ) / (x * sigma * sqrt(2 * th.pi))
    return th.sum(log_density, dim=[2, 3, 4])


def process_factor(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))
