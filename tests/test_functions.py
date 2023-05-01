from typing import Tuple

import pytest
import torch as th

from music_diffusion.networks.functions import normal_cdf


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
    sigma = th.rand(*x.size(), device=device)

    proba = normal_cdf(x, mu, sigma)

    assert len(proba.size()) == 2
    assert proba.size(0) == batch_size
    assert proba.size(1) == step_batch_size

    assert th.all(
        th.logical_and(th.ge(th.tensor(0), proba), th.le(proba, th.tensor(1)))
    )
