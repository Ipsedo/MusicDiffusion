from math import sqrt

import torch as th
import torch.distributions


def smart_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    value_range = max_value - min_value

    x = (x - min_value) / value_range % 1.0
    x[x < 0.0] += 1.0

    x = x * value_range + min_value

    return x


def bound_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    return th.clip(x, min_value, max_value)


def normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    b, t, c, w, h = x.size()
    dist = torch.distributions.Normal(mu.flatten(), sigma.flatten())
    density: th.Tensor = dist.log_prob(x.flatten())

    density = density.view(b, t, c, w, h)
    density = density.flatten(2, -1).sum(-1).exp()

    return density


def log_normal_pdf(x: th.Tensor, mu: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    log_density = th.exp(
        -th.pow(th.log(x) - mu, 2.0) / (2.0 * th.pow(sigma, 2.0))
    ) / (x * sigma * sqrt(2 * th.pi))

    return th.sum(log_density, dim=[2, 3, 4])


def process_factor(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))
