# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from music_diffusion.networks.functions import normal_cdf, normal_log_prob


@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_normal_log_prob(
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: Tuple[int, int],
    use_cuda: bool,
) -> None:

    device = "cuda" if use_cuda else "cpu"

    x = th.randn(
        batch_size, step_batch_size, channels, *img_sizes, device=device
    )
    mu = th.randn(*x.size(), device=device)
    sigma = th.rand(*x.size(), device=device) + 1e-8

    proba = normal_log_prob(x, mu, sigma)

    assert len(proba.size()) == 5
    assert proba.size(0) == batch_size
    assert proba.size(1) == step_batch_size
    assert proba.size(2) == channels
    assert proba.size(3) == img_sizes[0]
    assert proba.size(4) == img_sizes[1]

    # assert th.all(
    #    th.logical_and(th.ge(proba, th.tensor(0.)),
    #    th.le(proba, th.tensor(1.)))
    # )
    # assert th.all(
    #    th.le(proba, th.tensor(0))
    # )


@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_normal_cdf(
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: Tuple[int, int],
    use_cuda: bool,
) -> None:

    device = "cuda" if use_cuda else "cpu"

    x = th.randn(
        batch_size, step_batch_size, channels, *img_sizes, device=device
    )
    mu = th.randn(*x.size(), device=device)
    sigma = th.rand(*x.size(), device=device) + 1e-8

    proba = normal_cdf(x, mu, sigma)

    assert len(proba.size()) == 5
    assert proba.size(0) == batch_size
    assert proba.size(1) == step_batch_size
    assert proba.size(2) == channels
    assert proba.size(3) == img_sizes[0]
    assert proba.size(4) == img_sizes[1]

    assert th.all(
        th.logical_and(
            th.ge(proba, th.tensor(0.0)), th.le(proba, th.tensor(1.0))
        ),
    )
