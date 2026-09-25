import pytest
import torch as th

from music_diffusion.networks import Denoiser, Noiser, TimeUNet

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

    x_t, eps = noiser(x_0, t)

    __inner_check_size(x_t)
    __inner_check_size(eps)

    post_mu, post_var = noiser.posterior(x_t, x_0, t)

    __inner_check_size(post_mu)

    assert post_var.size() == (batch_size, step_batch_size, 1, 1, 1)
    assert th.all(th.gt(post_var, 0))


@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser(
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

    denoiser = Denoiser(steps, time_size, [(in_channels, 8), (8, 16)], [2, 4])

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

    eps, v = denoiser(x_t, t)

    __inner_check_size(eps)
    __inner_check_size(v)

    prior_mu, prior_var = denoiser.prior(x_t, t, eps, v)

    __inner_check_size(prior_mu)

    __inner_check_size(prior_var)
    assert th.all(th.gt(prior_var, 0.0))

    x_t = th.randn(
        batch_size,
        in_channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.sample(x_t)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])

    x_0 = denoiser.fast_sample(x_t, steps // 2)

    assert x_0.size() == (batch_size, in_channels, img_sizes[0], img_sizes[1])

    # test with batch size == 1
    denoiser.eval()

    x_t = th.randn(
        1,
        in_channels,
        *img_sizes,
        device=device,
    )

    # normal sample
    x_0 = denoiser.sample(x_t)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])

    # fast sample
    x_0 = denoiser.fast_sample(x_t, steps // 2)

    assert x_0.size() == (1, in_channels, img_sizes[0], img_sizes[1])


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("size", [(32, 32), (16, 32)])
@pytest.mark.parametrize(
    "channels",
    [[(2, 8), (8, 16), (16, 32)], [(4, 8), (8, 32), (32, 16)]],
)
@pytest.mark.parametrize(
    "group_norm_nums",
    [[2, 4, 2], [1, 2, 4]],
)
@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("step_batch_size", [1, 2])
def test_unet(
    batch_size: int,
    size: tuple[int, int],
    channels: list[tuple[int, int]],
    group_norm_nums: list[int],
    steps: int,
    time_size: int,
    step_batch_size: int,
    device: th.device,
) -> None:
    def __inner_check_size(tensor: th.Tensor) -> None:
        check_size(tensor, batch_size, step_batch_size, channels[0][0], size)

    unet = TimeUNet(
        channels,
        group_norm_nums,
        time_size,
        steps,
    )

    unet.to(device)
    unet.eval()

    x_t = th.randn(
        batch_size,
        step_batch_size,
        channels[0][0],
        *size,
        device=device,
    )
    t = th.randint(
        0,
        steps,
        (batch_size, step_batch_size),
        device=device,
    )

    eps, v = unet(x_t, t)

    __inner_check_size(eps)
    __inner_check_size(v)
