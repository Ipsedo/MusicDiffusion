# pylint: disable=duplicate-code

import pytest
import torch as th

from music_diffusion.networks import Denoiser, Noiser


@pytest.mark.parametrize("steps", [4, 16])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_velocity_prior_matches_posterior(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    device: th.device,
) -> None:
    in_channels = 2
    img_sizes = (16, 16)

    noiser = Noiser(steps)
    denoiser = Denoiser(steps, 2, [(in_channels, 8), (8, 16)], [2, 4], 2)

    noiser.to(device)
    denoiser.to(device)

    x_0 = th.rand(batch_size, in_channels, *img_sizes, device=device)
    t = th.randint(
        0,
        steps,
        (batch_size, step_batch_size),
        device=device,
    )

    x_t, v = noiser(x_0, t)

    post_mu, _ = noiser.posterior(x_t, x_0, t)
    prior_mu, _ = denoiser.prior(x_t, t, v, th.zeros_like(v))

    # true velocity => denoiser prior mean == noiser posterior mean
    assert th.allclose(prior_mu, post_mu, atol=1e-5)
