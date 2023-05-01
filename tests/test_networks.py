# -*- coding: utf-8 -*-
from typing import List, Tuple

import pytest
import torch as th

from music_diffusion.networks import Denoiser, Noiser, TimeUNet


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

    assert len(x_t.size()) == 5
    assert x_t.size(0) == batch_size
    assert x_t.size(1) == step_batch_size
    assert x_t.size(2) == channels
    assert x_t.size(3) == img_sizes[0]
    assert x_t.size(4) == img_sizes[1]

    assert len(eps.size()) == 5
    assert eps.size(0) == batch_size
    assert eps.size(1) == step_batch_size
    assert eps.size(2) == channels
    assert eps.size(3) == img_sizes[0]
    assert eps.size(4) == img_sizes[1]

    t_prev = t - 1
    t_prev[t_prev < 0] = 0

    x_t_prev, eps_same = noiser(x_0, t_prev, eps)

    assert th.all(th.eq(eps_same, eps))

    posterior = noiser.posterior(x_t_prev, x_t, x_0, t)

    assert len(posterior.size()) == 2
    assert posterior.size(0) == batch_size
    assert posterior.size(1) == step_batch_size

    assert th.all(
        th.logical_and(
            th.ge(th.tensor(0), posterior), th.le(posterior, th.tensor(1))
        )
    )


@pytest.mark.parametrize("steps", [10, 20])
@pytest.mark.parametrize("step_batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("norm_groups", [1, 2])
@pytest.mark.parametrize("img_sizes", [(16, 16), (8, 16)])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser(
    steps: int,
    step_batch_size: int,
    batch_size: int,
    channels: int,
    norm_groups: int,
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
        [(8, 8), (8, 16), (16, 32)],
        [False, False, True],
        2,
        norm_groups,
    )

    denoiser.eval()

    if use_cuda:
        denoiser.cuda()
        device = "cuda"
    else:
        device = "cpu"

    x_t = th.randn(
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

    eps, v = denoiser(x_t, t)

    assert len(eps.size()) == 5
    assert eps.size(0) == batch_size
    assert eps.size(1) == step_batch_size
    assert eps.size(2) == channels
    assert eps.size(3) == img_sizes[0]
    assert eps.size(4) == img_sizes[1]

    assert len(v.size()) == 5
    assert v.size(0) == batch_size
    assert v.size(1) == step_batch_size
    assert v.size(2) == channels
    assert v.size(3) == img_sizes[0]
    assert v.size(4) == img_sizes[1]

    x_t_prev = th.randn(*x_t.size())

    prior = denoiser.prior(x_t_prev, x_t, t, eps, v)

    assert len(prior.size()) == 2
    assert prior.size(0) == batch_size
    assert prior.size(1) == step_batch_size
    assert th.all(
        th.logical_and(th.ge(th.tensor(0), prior), th.le(prior, th.tensor(1)))
    )

    x_t = th.randn(
        batch_size,
        channels,
        *img_sizes,
        device=device,
    )

    x_0 = denoiser.sample(x_t)

    assert len(x_0.size()) == 4
    assert x_0.size(0) == batch_size
    assert x_0.size(1) == channels
    assert x_0.size(2) == img_sizes[0]
    assert x_0.size(3) == img_sizes[1]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("norm_groups", [1, 2])
@pytest.mark.parametrize("size", [(16, 16), (8, 16)])
@pytest.mark.parametrize(
    "hidden_channels",
    [[(16, 8), (8, 16), (16, 32)], [(16, 8), (8, 40), (40, 16)]],
)
@pytest.mark.parametrize(
    "use_attentions", [[True, False, True], [False, True, False]]
)
@pytest.mark.parametrize("attention_heads", [1, 2])
@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("time_size", [2, 4])
@pytest.mark.parametrize("nb_steps", [1, 2])
def test_unet(
    batch_size: int,
    channels: int,
    norm_groups: int,
    size: Tuple[int, int],
    hidden_channels: List[Tuple[int, int]],
    use_attentions: List[bool],
    attention_heads: int,
    steps: int,
    time_size: int,
    nb_steps: int,
    use_cuda: bool,
) -> None:
    unet = TimeUNet(
        channels,
        channels,
        hidden_channels,
        use_attentions,
        attention_heads,
        time_size,
        steps,
        norm_groups,
    )

    unet.eval()

    if use_cuda:
        unet.cuda()
        device = "cuda"
    else:
        device = "cpu"

    x_t = th.randn(
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

    eps, v = unet(x_t, t)

    assert len(eps.size()) == 5
    assert eps.size(0) == batch_size
    assert eps.size(1) == nb_steps
    assert eps.size(2) == channels
    assert eps.size(3) == size[0]
    assert eps.size(4) == size[1]

    assert len(v.size()) == 5
    assert v.size(0) == batch_size
    assert v.size(1) == nb_steps
    assert v.size(2) == channels
    assert v.size(3) == size[0]
    assert v.size(4) == size[1]
