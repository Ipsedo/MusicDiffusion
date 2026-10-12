# pylint: disable=duplicate-code

import pytest
import torch as th

from music_diffusion.networks import TimeUNet

from .check_size import check_size


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
@pytest.mark.parametrize("z_size", [2, 3])
def test_unet(
    batch_size: int,
    size: tuple[int, int],
    channels: list[tuple[int, int]],
    group_norm_nums: list[int],
    steps: int,
    time_size: int,
    step_batch_size: int,
    z_size: int,
    device: th.device,
) -> None:
    def __inner_check_size(tensor: th.Tensor) -> None:
        check_size(tensor, batch_size, step_batch_size, channels[0][0], size)

    unet = TimeUNet(channels, group_norm_nums, time_size, steps, z_size)

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

    z = th.randn(batch_size, z_size, device=device)

    v_theta, var_interp = unet(x_t, t, z)

    __inner_check_size(v_theta)
    __inner_check_size(var_interp)


def test_unet_z_changes_output(device: th.device) -> None:
    th.manual_seed(0)

    unet = TimeUNet([(2, 4)], [2], 2, 2, z_size=3)
    unet.to(device)
    unet.eval()

    # the last FiLM layer is zero-initialized : perturb it so z has an effect
    for p in unet.parameters():
        p.data.add_(th.randn_like(p) * 1e-1)

    x_t = th.randn(1, 1, 2, 8, 8, device=device)
    t = th.zeros(1, 1, dtype=th.long, device=device)

    v_a, _ = unet(x_t, t, th.zeros(1, 3, device=device))
    v_b, _ = unet(x_t, t, th.ones(1, 3, device=device))

    assert not th.allclose(v_a, v_b)
