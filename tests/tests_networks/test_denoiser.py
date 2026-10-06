# pylint: disable=duplicate-code

import pytest
import torch as th

from music_diffusion.networks.denoiser import Denoiser

from .check_size import check_size


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser_forward(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    device: th.device,
) -> None:
    in_channels = 2

    def __inner_check_size(tensor: th.Tensor) -> None:
        check_size(tensor, batch_size, step_batch_size, in_channels, img_sizes)

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        batch_size,
        step_batch_size,
        in_channels,
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

    v_theta, var_interp = denoiser(x_t, t)

    __inner_check_size(v_theta)
    __inner_check_size(var_interp)

    prior_mu, prior_var = denoiser.prior(x_t, t, v_theta, var_interp)

    __inner_check_size(prior_mu)

    __inner_check_size(prior_var)
    assert th.all(th.gt(prior_var, 0.0))


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser_sample(
    steps: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        batch_size,
        in_channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.sample(x_t)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser_single_sample(
    steps: int,
    img_sizes: tuple[int, int],
    time_size: int,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        1,
        in_channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.sample(x_t)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser_fast_sample(
    steps: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        batch_size,
        in_channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.fast_sample(x_t, steps // 2)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser_single_fast_sample(
    steps: int,
    img_sizes: tuple[int, int],
    time_size: int,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        1,
        in_channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.fast_sample(x_t, steps // 2)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])
