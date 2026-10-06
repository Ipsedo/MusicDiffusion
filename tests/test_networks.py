from typing import Literal

import pytest
import torch as th
from torch import nn

from music_diffusion.networks import Denoiser, Noiser, TimeUNet
from music_diffusion.networks.convolutions import (
    StrideConvBlock,
    TimeConvBlock,
)
from music_diffusion.networks.liquid import LiquidRecurrent
from music_diffusion.networks.time import TimeBypass

from .check_size import check_size


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("input_channels", [2, 3])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_time_bypass(
    batch_size: int,
    step_batch_size: int,
    input_channels: int,
    output_channels: int,
    device: th.device,
) -> None:
    sizes = (4, 4)

    conv2d_time_bypass = TimeBypass(
        nn.Conv2d(input_channels, output_channels, 3, 1, 1)
    )
    conv2d_time_bypass.to(device)

    x = th.randn(
        batch_size, step_batch_size, input_channels, *sizes, device=device
    )

    out = conv2d_time_bypass(x)

    check_size(out, batch_size, step_batch_size, output_channels, sizes)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("group_norm_num", [1, 2])
@pytest.mark.parametrize("up_or_down", ["up", "down"])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
def test_time_stride_conv_block(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    group_norm_num: int,
    up_or_down: Literal["up", "down"],
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    out_sizes = (
        (img_sizes[0] // 2, img_sizes[1] // 2)
        if up_or_down == "down"
        else (img_sizes[0] * 2, img_sizes[1] * 2)
    )

    time_conv_block = StrideConvBlock(
        in_channels, out_channels, group_norm_num, up_or_down
    )
    time_conv_block.to(device)

    x = th.randn(
        batch_size, in_channels, img_sizes[0], img_sizes[1], device=device
    )

    out = time_conv_block(x)

    assert out.size() == (batch_size, out_channels, out_sizes[0], out_sizes[1])


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("step_btch_size", [1, 2])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("group_norm_num", [1, 2])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
def test_time_conv_block(
    batch_size: int,
    step_btch_size: int,
    in_channels: int,
    out_channels: int,
    group_norm_num: int,
    time_size: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    time_conv_block = TimeConvBlock(
        in_channels, out_channels, group_norm_num, time_size
    )

    time_conv_block.to(device)

    x = th.randn(
        batch_size,
        step_btch_size,
        in_channels,
        img_sizes[0],
        img_sizes[1],
        device=device,
    )
    t_emb = th.randn(batch_size, step_btch_size, time_size)

    out = time_conv_block(x, t_emb)

    check_size(out, batch_size, step_btch_size, out_channels, img_sizes)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("input_size", [1, 2])
@pytest.mark.parametrize("output_size", [1, 2])
@pytest.mark.parametrize("neuron_number", [1, 2])
@pytest.mark.parametrize("unfolding_steps", [1, 2])
@pytest.mark.parametrize("time_steps", [1, 2])
def test_liquid(
    batch_size: int,
    input_size: int,
    output_size: int,
    neuron_number: int,
    unfolding_steps: int,
    time_steps: int,
    device: th.Device,
) -> None:
    ltc = LiquidRecurrent(
        neuron_number, input_size, output_size, unfolding_steps, nn.SiLU(), 1.0
    )
    ltc.to(device)

    x = th.randn(batch_size, time_steps, input_size, device=device)

    out = ltc(x)

    assert out.size() == (batch_size, time_steps, output_size)


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

    unet = TimeUNet(channels, group_norm_nums, time_size, steps)

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

    v_theta, var_interp = unet(x_t, t)

    __inner_check_size(v_theta)
    __inner_check_size(var_interp)


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
    denoiser = Denoiser(steps, 2, [(in_channels, 8), (8, 16)], [2, 4])

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
