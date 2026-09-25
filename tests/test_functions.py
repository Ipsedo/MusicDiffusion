import pytest
import torch as th

from music_diffusion.networks.functions import normal_cdf, normal_log_prob

from .check_size import check_size


def _get_x_mu_sigma(
    batch_size: int,
    step_batch_size: int,
    channels: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    x = th.randn(
        batch_size, step_batch_size, channels, *img_sizes, device=device
    )
    mu = th.randn(*x.size(), device=device)
    sigma = th.exp(th.randn(*x.size(), device=device) + 1e-8)

    return x, mu, sigma


@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_normal_log_prob(
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    x, mu, sigma = _get_x_mu_sigma(
        batch_size, step_batch_size, channels, img_sizes, device
    )

    log_proba = normal_log_prob(x, mu, sigma)

    check_size(log_proba, batch_size, step_batch_size, channels, img_sizes)


@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_normal_cdf(
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    x, mu, sigma = _get_x_mu_sigma(
        batch_size, step_batch_size, channels, img_sizes, device
    )

    proba = normal_cdf(x, mu, sigma)

    check_size(proba, batch_size, step_batch_size, channels, img_sizes)

    assert th.all(
        th.logical_and(
            th.ge(proba, th.tensor(0.0)), th.le(proba, th.tensor(1.0))
        ),
    )
