import pytest
import torch as th

from music_diffusion.data.primitive import simpson, trapezoid


@pytest.mark.parametrize("dx", [0.01, 0.02, 0.04])
def test_simpson(dx: float) -> None:
    start = -8.0
    end = 8.0

    steps = int((end - start) / dx)

    delta = 1e-2
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = simpson(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)


@pytest.mark.parametrize("dx", [0.01, 0.02, 0.04])
def test_trapezoid(dx: float) -> None:
    start = -8.0
    end = 8.0

    steps = int((end - start) / dx)

    delta = 1e-2
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = trapezoid(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)
