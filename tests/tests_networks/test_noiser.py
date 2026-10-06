# pylint: disable=duplicate-code

import pytest
import torch as th

from music_diffusion.networks import Noiser

from .check_size import check_size


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_noiser(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    def __inner_check_size(tensor: th.Tensor) -> None:
        check_size(tensor, batch_size, step_batch_size, channels, img_sizes)

    noiser = Noiser(steps)
    noiser.to(device)

    x_0 = th.randn(
        batch_size,
        channels,
        img_sizes[0],
        img_sizes[1],
        device=device,
    )
    t = th.randint(
        0,
        steps,
        (batch_size, step_batch_size),
        device=device,
    )

    x_t, v = noiser(x_0, t)

    __inner_check_size(x_t)
    __inner_check_size(v)

    post_mu, post_var = noiser.posterior(x_t, x_0, t)

    __inner_check_size(post_mu)

    assert post_var.size() == (batch_size, step_batch_size, 1, 1, 1)
    assert th.all(th.gt(post_var, 0))
