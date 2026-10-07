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
@pytest.mark.parametrize("z_size", [0, 3])
def test_denoiser_forward(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    z_size: int,
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
        z_size,
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

    x_ref = th.randn(batch_size, in_channels, *img_sizes, device=device)
    z = denoiser.encode(x_ref)

    assert z.size() == (batch_size, z_size)

    v_theta, var_interp = denoiser(x_t, t, z)

    __inner_check_size(v_theta)
    __inner_check_size(var_interp)

    # z defaults to the null condition
    v_theta_null, _ = denoiser(x_t, t)
    __inner_check_size(v_theta_null)

    prior_mu, prior_var = denoiser.prior(x_t, t, v_theta, var_interp)

    __inner_check_size(prior_mu)

    __inner_check_size(prior_var)
    assert th.all(th.gt(prior_var, 0.0))


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("z_size", [0, 3])
@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_denoiser_sample(
    steps: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    z_size: int,
    guidance_scale: float,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
        z_size,
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        batch_size,
        in_channels,
        *img_sizes,
        device=device,
    )

    z = denoiser.encode(th.randn_like(x_t))

    x_0 = denoiser.sample(x_t, z, guidance_scale)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("z_size", [0, 3])
@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_denoiser_single_sample(
    steps: int,
    img_sizes: tuple[int, int],
    time_size: int,
    z_size: int,
    guidance_scale: float,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
        z_size,
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        1,
        in_channels,
        *img_sizes,
        device=device,
    )

    z = denoiser.encode(th.randn_like(x_t))

    x_0 = denoiser.sample(x_t, z, guidance_scale)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("z_size", [0, 3])
@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_denoiser_fast_sample(
    steps: int,
    batch_size: int,
    img_sizes: tuple[int, int],
    time_size: int,
    z_size: int,
    guidance_scale: float,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
        z_size,
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        batch_size,
        in_channels,
        *img_sizes,
        device=device,
    )

    z = denoiser.encode(th.randn_like(x_t))

    x_0 = denoiser.fast_sample(x_t, steps // 2, z, guidance_scale)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("z_size", [0, 3])
@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_denoiser_single_fast_sample(
    steps: int,
    img_sizes: tuple[int, int],
    time_size: int,
    z_size: int,
    guidance_scale: float,
    device: th.device,
) -> None:
    in_channels = 2

    denoiser = Denoiser(
        steps,
        time_size,
        [(in_channels, 8), (8, 16)],
        [2, 4],
        z_size,
    )

    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(
        1,
        in_channels,
        *img_sizes,
        device=device,
    )

    z = denoiser.encode(th.randn_like(x_t))

    x_0 = denoiser.fast_sample(x_t, steps // 2, z, guidance_scale)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])


def test_denoiser_null_condition(device: th.device) -> None:
    denoiser = Denoiser(2, 2, [(2, 4)], [2], z_size=3)
    denoiser.to(device)

    z_null = denoiser.null_condition(4, device)

    assert z_null.size() == (4, 3)
    assert th.all(th.eq(z_null, 0.0))

    # without z the encoder is absent and z is empty
    denoiser_no_z = Denoiser(2, 2, [(2, 4)], [2], z_size=0)
    denoiser_no_z.to(device)

    assert denoiser_no_z.encode(
        th.randn(4, 2, 8, 8, device=device)
    ).size() == (
        4,
        0,
    )


def test_denoiser_guidance_matches_unguided_at_one(
    device: th.device,
) -> None:
    th.manual_seed(0)

    denoiser = Denoiser(4, 2, [(2, 4)], [2], z_size=3)
    denoiser.to(device)
    denoiser.eval()

    x_t = th.randn(2, 2, 8, 8, device=device)
    z = denoiser.encode(th.randn_like(x_t))

    th.manual_seed(1)
    x_a = denoiser.sample(x_t, z, guidance_scale=1.0)
    th.manual_seed(1)
    x_b = denoiser.sample(x_t, z, guidance_scale=1.0)

    assert th.allclose(x_a, x_b)
