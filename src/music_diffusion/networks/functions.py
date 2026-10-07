import torch as th


def select_time_scheduler(factor: th.Tensor, t: th.Tensor) -> th.Tensor:
    b, s = t.size()
    factor = factor[t.flatten(), None, None, None]
    return th.unflatten(factor, 0, (b, s))


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


def drop_condition(z: th.Tensor, p: float) -> th.Tensor:
    assert 0.0 <= p <= 1.0

    if p == 0.0:
        return z

    keep = th.rand(z.size(0), device=z.device) >= p

    return z * keep[:, None].to(z.dtype)
