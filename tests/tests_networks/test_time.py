# pylint: disable=duplicate-code

import pytest
import torch as th
from torch import nn

from music_diffusion.networks.time import (
    SinusoidTimeEmbedding,
    TimeBypass,
    TimeToScaleShift,
)

from .check_size import check_size


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("nb_steps", [2, 3])
@pytest.mark.parametrize("time_embedding_size", [2, 4])
def test_time_embedding(
    batch_size: int,
    step_batch_size: int,
    nb_steps: int,
    time_embedding_size: int,
    device: th.device,
) -> None:
    time_embedding = SinusoidTimeEmbedding(nb_steps, time_embedding_size)
    time_embedding.to(device)

    x = th.randint(0, nb_steps, (batch_size, step_batch_size), device=device)

    emb = time_embedding(x)

    assert emb.size() == (batch_size, step_batch_size, time_embedding_size)


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
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("time_size", [2, 3])
def test_time_to_scale_shift(
    batch_size: int,
    step_batch_size: int,
    channels: int,
    time_size: int,
    device: th.Device,
) -> None:
    time_to_scale_shift = TimeToScaleShift(channels, time_size)
    time_to_scale_shift.to(device)

    time_embedding = th.randn(
        batch_size, step_batch_size, time_size, device=device
    )

    scale, shift = time_to_scale_shift(time_embedding)

    assert scale.size() == (batch_size, step_batch_size, channels, 1, 1)
    assert shift.size() == (batch_size, step_batch_size, channels, 1, 1)
