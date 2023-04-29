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

    t_minus_one = t - 1
    t_minus_one[t_minus_one < 0] = 0

    x_noised_minus, eps = noiser(x, t_minus_one, eps)

    q = noiser.posterior(x_noised_minus, x_noised, x, t)

    assert len(q.size()) == 2
    assert q.size(0) == batch_size
    assert q.size(1) == step_batch_size


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

    eps, v = denoiser(x, t)

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

    x_t_minus = x.clone()
    t_minus = t - 1
    t_minus[t_minus < 0] = 0

    prior = denoiser.prior(x, x_t_minus, t_minus, eps, v)

    assert len(prior.size()) == 2
    assert prior.size(0) == batch_size
    assert prior.size(1) == step_batch_size

    x = th.randn(
        batch_size,
        channels,
        *img_sizes,
        device=device,
    )

    pred = denoiser.sample(x)

    assert len(pred.size()) == 4
    assert pred.size(0) == batch_size
    assert pred.size(1) == channels
    assert pred.size(2) == img_sizes[0]
    assert pred.size(3) == img_sizes[1]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("norm_groups", [1, 2])
@pytest.mark.parametrize("size", [(16, 16), (8, 16)])
@pytest.mark.parametrize(
    "hidden_channels",
    [[(16, 8), (8, 16), (16, 32)], [(16, 8), (8, 40), (40, 16)]],
)
@pytest.mark.parametrize(
    "use_attentions", [[True, True, True], [False, False, False]]
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

    o, _ = unet(x, t)

    assert len(o.size()) == 5
    assert o.size(0) == batch_size
    assert o.size(1) == nb_steps
    assert o.size(2) == channels
    assert o.size(3) == size[0]
    assert o.size(4) == size[1]
