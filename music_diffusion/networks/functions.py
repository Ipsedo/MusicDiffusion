from math import sqrt

import torch as th
from torch.distributions import Normal


def smart_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    value_range = max_value - min_value

    x = (x - min_value) / value_range % 1.0
    x[x < 0.0] += 1.0

    x = x * value_range + min_value

    return x


def bound_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    return th.clip(x, min_value, max_value)


def normal_cdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    b, t = x.size()[:2]

    dist = Normal(mu.flatten(), sigma.flatten())
    density: th.Tensor = dist.cdf(x.flatten()).view(b, t, -1).prod(-1)

    return density


def log_normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    log_density = th.exp(
        -th.pow(th.log(x) - mu, 2.0) / (2.0 * th.pow(sigma, 2.0))
    ) / (x * sigma * sqrt(2 * th.pi))

    return th.sum(log_density, dim=[2, 3, 4])


def select_time_scheduler(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))
