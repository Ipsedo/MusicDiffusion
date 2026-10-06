# pylint: disable=duplicate-code

from typing import Literal

import pytest
import torch as th

from music_diffusion.networks.convolutions import (
    OutChannelProj,
    StrideConvBlock,
    TimeConvBlock,
)

from .check_size import check_size


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("img_sizes", [(32, 32), (16, 32)])
def test_out_conv_block(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    img_sizes: tuple[int, int],
    device: th.device,
) -> None:
    out_conv_block = OutChannelProj(in_channels, out_channels)
    out_conv_block.to(device)

    x = th.randn(
        batch_size, in_channels, img_sizes[0], img_sizes[1], device=device
    )

    out = out_conv_block(x)

    assert out.size() == (batch_size, out_channels, img_sizes[0], img_sizes[1])


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
