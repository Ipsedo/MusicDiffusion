from typing import List, Tuple

import pytest
import torch as th

from music_diffusion_model.networks import Denoiser, Noiser, TimeUNet


@pytest.mark.parametrize("steps", [10, 20])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
def test_noiser(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: Tuple[int, int],
    use_cuda: bool,
) -> None:
    noiser = Noiser(steps, 1e-4, 2e-1)

    if use_cuda:
        noiser.cuda()
        device = "cuda"
    else:
        device = "cpu"

    x = th.randn(
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

    x_noised, eps = noiser(x, t)

    assert len(x_noised.size()) == 5
    assert x_noised.size(0) == batch_size
    assert x_noised.size(1) == step_batch_size
    assert x_noised.size(2) == channels
    assert x_noised.size(3) == img_sizes[0]
    assert x_noised.size(4) == img_sizes[1]

    assert len(eps.size()) == 5
    assert eps.size(0) == batch_size
    assert eps.size(1) == step_batch_size
    assert eps.size(2) == channels
    assert eps.size(3) == img_sizes[0]
    assert eps.size(4) == img_sizes[1]


@pytest.mark.parametrize("steps", [10, 20])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("img_sizes", [(32, 32), (32, 64)])
@pytest.mark.parametrize("time_size", [2, 3])
def test_denoiser(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    channels: int,
    img_sizes: Tuple[int, int],
    time_size: int,
    use_cuda: bool,
) -> None:
    denoiser = Denoiser(
        channels,
        steps,
        time_size,
        1e-4,
        2e-1,
        [(4, 8), (8, 16), (16, 32)],
    )

    denoiser.eval()

    if use_cuda:
        denoiser.cuda()
        device = "cuda"
    else:
        device = "cpu"

    x = th.randn(
        batch_size,
        step_batch_size,
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

    o = denoiser(x, t)

    assert len(o.size()) == 5
    assert o.size(0) == batch_size
    assert o.size(1) == step_batch_size
    assert o.size(2) == channels
    assert o.size(3) == img_sizes[0]
    assert o.size(4) == img_sizes[1]

    x = th.randn(
        batch_size,
        channels,
        *img_sizes,
        device=device,
    )

    o = denoiser.sample(x)

    assert len(o.size()) == 4
    assert o.size(0) == batch_size
    assert o.size(1) == channels
    assert o.size(2) == img_sizes[0]
    assert o.size(3) == img_sizes[1]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("size", [(32, 32), (32, 64)])
@pytest.mark.parametrize(
    "hidden_channels",
    [[(4, 8), (8, 16), (16, 32)], [(18, 5), (5, 43), (43, 3)]],
)
@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("time_size", [2, 3])
@pytest.mark.parametrize("nb_steps", [1, 2])
def test_unet(
    batch_size: int,
    channels: int,
    size: Tuple[int, int],
    hidden_channels: List[Tuple[int, int]],
    steps: int,
    time_size: int,
    nb_steps: int,
    use_cuda: bool,
) -> None:
    unet = TimeUNet(
        channels,
        channels,
        hidden_channels,
        time_size,
        steps,
    )

    unet.eval()

    if use_cuda:
        unet.cuda()
        device = "cuda"
    else:
        device = "cpu"

    x = th.randn(
        batch_size,
        nb_steps,
        channels,
        *size,
        device=device,
    )
    t = th.randint(
        0,
        steps,
        (batch_size, nb_steps),
        device=device,
    )

    o = unet(x, t)

    assert len(o.size()) == 5
    assert o.size(0) == batch_size
    assert o.size(1) == nb_steps
    assert o.size(2) == channels
    assert o.size(3) == size[0]
    assert o.size(4) == size[1]
