# -*- coding: utf-8 -*-
import torch as th
from torch.nn import functional as th_f


def simpson(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    sizes = derivative.size()
    n = sizes[dim]

    evens = th.arange(0, n, 2)
    odds = th.arange(1, n, 2)

    even_derivative = th.index_select(derivative, dim, evens)
    odd_derivative = th.index_select(derivative, dim, odds)

    shift_odd_derivative = th_f.pad(
        odd_derivative,
        [
            p
            for d in reversed(range(len(sizes)))
            for p in [1 if d == dim else 0, 0]
        ],
        "constant",
        0,
    )

    even_primitive = first_primitive + dx / 3 * (
        (
            2 * even_derivative
            + 4
            * th.index_select(
                shift_odd_derivative,
                dim=dim,
                index=th.arange(0, even_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - th.select(even_derivative, dim, 0).unsqueeze(dim)
        - even_derivative
    )

    odd_primitive = (dx / 3) * (
        (
            2 * odd_derivative
            + 4
            * th.index_select(
                even_derivative,
                dim=dim,
                index=th.arange(0, odd_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - 4 * th.select(even_derivative, dim, 0).unsqueeze(dim)
        - th.select(odd_derivative, dim, 0).unsqueeze(dim)
        - odd_derivative
    )

    odd_primitive += first_primitive + dx / 12 * (
        5 * th.select(derivative, dim, 0)
        + 8 * th.select(derivative, dim, 1)
        - th.select(derivative, dim, 2)
    ).unsqueeze(dim)

    primitive = th.zeros_like(derivative)

    view = [-1 if i == dim else 1 for i in range(len(sizes))]
    repeat = [1 if i == dim else s for i, s in enumerate(sizes)]
    evens = evens.view(*view).repeat(*repeat)
    odds = odds.view(*view).repeat(*repeat)

    primitive.scatter_(dim, evens, even_primitive)
    primitive.scatter_(dim, odds, odd_primitive)

    return primitive


def trapezoid(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    return first_primitive + dx * (
        derivative.cumsum(dim=dim)
        - derivative / 2.0
        - th.select(derivative, dim, 0).unsqueeze(dim) / 2.0
    )
