# pylint: disable=duplicate-code

import pytest
import torch as th
from torch import nn

from music_diffusion.networks.liquid import LiquidRecurrent


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
