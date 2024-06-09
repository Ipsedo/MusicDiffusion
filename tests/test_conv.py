# -*- coding: utf-8 -*-
import pytest
import torch as th

from music_diffusion.networks.conv1d import CausalConv1d, CausalConvTr1d


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [1, 2, 3])
@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
def test_causal_conv_1d(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> None:
    causal_conv = CausalConv1d(
        in_channels, out_channels, kernel_size, stride, dilation
    )

    x = th.randn(batch_size, in_channels, size)
    o = causal_conv(x)

    assert len(o.size()) == 3
    assert o.size(0) == batch_size
    assert o.size(1) == out_channels
    assert o.size(2) == size // stride


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_channels", [1, 2, 3])
@pytest.mark.parametrize("out_channels", [1, 2, 3])
@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("kernel_size", [5, 6, 7])
@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_causal_conv_tr_1d(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> None:
    causal_conv = CausalConvTr1d(
        in_channels, out_channels, kernel_size, stride, dilation
    )

    x = th.randn(batch_size, in_channels, size)
    o = causal_conv(x)

    assert len(o.size()) == 3
    assert o.size(0) == batch_size
    assert o.size(1) == out_channels
    assert o.size(2) == size * stride
