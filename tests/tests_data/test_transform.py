# pylint: disable=duplicate-code
import pytest
import torch as th
from torchvision.transforms import Compose

from music_diffusion.data import (
    ChangeType,
    ChannelMinMaxNorm,
    InverseRangeChange,
    RangeChange,
)
from music_diffusion.data.transform import ImgTransform

# ChannelMinMaxNorm


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sizes", [(8, 16), (16, 16)])
def test_channel_min_max_norm(
    batch_size: int, channels: int, sizes: tuple[int, int]
) -> None:
    x = th.randn(batch_size, channels, *sizes)

    out = ChannelMinMaxNorm()(x)

    assert out.size() == x.size()
    assert out.dtype == x.dtype

    out_min = th.amin(out, dim=(-2, -1))
    out_max = th.amax(out, dim=(-2, -1))

    assert th.all(th.eq(out_min, 0.0))
    assert th.allclose(out_max, th.ones_like(out_max), atol=1e-6)


@pytest.mark.parametrize("value", [0.0, 1.0, -3.5])
def test_channel_min_max_norm_constant_channel(value: float) -> None:
    x = th.full((1, 1, 4, 4), value)

    out = ChannelMinMaxNorm()(x)

    assert not th.any(th.isnan(out))
    assert th.all(th.eq(out, 0.0))


def test_channel_min_max_norm_per_channel() -> None:
    # channel 0 in [0, 1], channel 1 in [0, 100]
    small = th.rand(1, 1, 8, 8)
    big = th.rand(1, 1, 8, 8) * 100.0
    x = th.cat([small, big], dim=1)

    out = ChannelMinMaxNorm()(x)

    # a global normalisation would leave channel 0 far below 1
    assert th.isclose(th.amax(out[0, 0]), th.tensor(1.0), atol=1e-6)
    assert th.isclose(th.amax(out[0, 1]), th.tensor(1.0), atol=1e-6)
    assert th.eq(th.amin(out[0, 0]), 0.0)
    assert th.eq(th.amin(out[0, 1]), 0.0)


@pytest.mark.parametrize("sizes", [(8, 8, 8), (1, 1, 8, 8, 8), (8,)])
def test_channel_min_max_norm_wrong_dim(sizes: tuple[int, ...]) -> None:
    with pytest.raises(AssertionError):
        ChannelMinMaxNorm()(th.randn(*sizes))


def test_channel_min_max_norm_epsilon() -> None:
    # range is exactly 1 -> divisor is 1 + epsilon = 2
    x = th.tensor([[[[0.0, 1.0], [0.5, 0.25]]]])

    out = ChannelMinMaxNorm(epsilon=1.0)(x)

    assert th.isclose(th.amax(out), th.tensor(0.5))
    assert th.allclose(out, x / 2.0)


# InverseRangeChange / RangeChange

BOUNDS = [(-1.0, 1.0), (0.0, 255.0), (-3.0, 7.0)]


@pytest.mark.parametrize("bounds", BOUNDS)
def test_inverse_range_change(bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    tr = InverseRangeChange(lo, hi)

    assert th.equal(tr(th.tensor([lo])), th.tensor([0.0]))
    assert th.equal(tr(th.tensor([hi])), th.tensor([1.0]))
    assert th.equal(tr(th.tensor([(lo + hi) / 2.0])), th.tensor([0.5]))


@pytest.mark.parametrize("bounds", BOUNDS)
def test_range_change(bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    tr = RangeChange(lo, hi)

    assert th.equal(tr(th.tensor([0.0])), th.tensor([lo]))
    assert th.equal(tr(th.tensor([1.0])), th.tensor([hi]))
    assert th.equal(tr(th.tensor([0.5])), th.tensor([(lo + hi) / 2.0]))


@pytest.mark.parametrize("bounds", BOUNDS)
@pytest.mark.parametrize("sizes", [(16,), (2, 3, 4, 5)])
def test_range_change_round_trip(
    bounds: tuple[float, float], sizes: tuple[int, ...]
) -> None:
    lo, hi = bounds
    x = th.randn(*sizes) * 10.0

    forward = RangeChange(lo, hi)
    inverse = InverseRangeChange(lo, hi)

    assert th.allclose(forward(inverse(x)), x, atol=1e-5)
    assert th.allclose(inverse(forward(x)), x, atol=1e-5)


@pytest.mark.parametrize("sizes", [(16,), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", [th.float32, th.float64])
def test_range_change_shape_dtype(
    sizes: tuple[int, ...], dtype: th.dtype
) -> None:
    x = th.rand(*sizes, dtype=dtype)

    for tr in (RangeChange(-1.0, 1.0), InverseRangeChange(-1.0, 1.0)):
        out = tr(x)

        assert out.size() == x.size()
        assert out.dtype == dtype


# ChangeType


@pytest.mark.parametrize("dtype", [th.float16, th.float64, th.uint8, th.long])
@pytest.mark.parametrize("sizes", [(16,), (2, 3, 4, 5)])
def test_change_type(dtype: th.dtype, sizes: tuple[int, ...]) -> None:
    x = th.rand(*sizes) * 10.0

    out = ChangeType(dtype)(x)

    assert out.dtype == dtype
    assert out.size() == x.size()


def test_change_type_truncates() -> None:
    # `Tensor.to(uint8)` truncates toward zero, it does NOT round:
    # 254.7 -> 254 (and not 255), 0.9 -> 0
    out = ChangeType(th.uint8)(th.tensor([254.7, 0.9, 1.0]))

    assert th.equal(out, th.tensor([254, 0, 1], dtype=th.uint8))


# Composition used by the Saver


def _saver_transform() -> Compose:
    return Compose(
        [
            InverseRangeChange(-1, 1),
            RangeChange(0.0, 255.0),
            ChangeType(th.uint8),
        ]
    )


def test_saver_compose() -> None:
    x = th.rand(2, 2, 8, 8) * 2.0 - 1.0

    out = _saver_transform()(x)

    assert out.dtype == th.uint8
    assert out.size() == x.size()
    assert th.all(th.ge(out, 0))
    assert th.all(th.le(out, 255))


def test_saver_compose_bounds() -> None:
    out = _saver_transform()(th.tensor([-1.0, 0.0, 1.0]))

    # 0.0 maps to 127.5, then truncated to 127 by the uint8 cast
    assert th.equal(out, th.tensor([0, 127, 255], dtype=th.uint8))


def test_saver_compose_out_of_range() -> None:
    # The Saver pipeline does not clip before the uint8 cast: a value
    # slightly outside [-1, 1] lands outside [0, 255] as a float. The
    # final float -> uint8 cast of such a value is undefined behaviour
    # in C (observed on CPU: truncation then wrap modulo 256, e.g.
    # 1.5 -> 318.75 -> 62), so only the un-clipped float stage is
    # asserted here, plus the output dtype.
    unclipped = Compose([InverseRangeChange(-1, 1), RangeChange(0.0, 255.0)])

    out_float = unclipped(th.tensor([1.5, -2.0, 2.0]))
    assert th.allclose(out_float, th.tensor([318.75, -127.5, 382.5]))
    assert th.any(th.gt(out_float, 255.0))
    assert th.any(th.lt(out_float, 0.0))

    out = _saver_transform()(th.tensor([1.5, -2.0, 2.0]))
    assert out.dtype == th.uint8


# ImgTransform


def test_img_transform_abstract() -> None:
    with pytest.raises(TypeError):
        ImgTransform()  # type: ignore[abstract]  # pylint: disable=E0110
