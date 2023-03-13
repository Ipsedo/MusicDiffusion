import torch as th


def smart_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    value_range = max_value - min_value

    x = (x - min_value) / value_range % 1.0
    x[x < 0.0] += 1.0

    x = x * value_range + min_value

    return x


def bound_clip(x: th.Tensor, min_value: float, max_value: float) -> th.Tensor:
    return th.clip(x, min_value, max_value)
