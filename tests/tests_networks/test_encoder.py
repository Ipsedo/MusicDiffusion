import pytest
import torch as th

from music_diffusion.networks import ChunkEncoder


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("size", [(32, 32), (16, 32)])
@pytest.mark.parametrize(
    "channels",
    [[(2, 8), (8, 16), (16, 32)], [(2, 4)]],
)
@pytest.mark.parametrize("z_size", [1, 8])
def test_encoder(
    batch_size: int,
    size: tuple[int, int],
    channels: list[tuple[int, int]],
    z_size: int,
    device: th.device,
) -> None:
    group_norm_nums = [2] * len(channels)

    encoder = ChunkEncoder(channels, group_norm_nums, z_size)
    encoder.to(device)
    encoder.eval()

    assert encoder.z_size == z_size

    x_0 = th.randn(batch_size, channels[0][0], *size, device=device)

    z = encoder(x_0)

    assert z.size() == (batch_size, z_size)
    assert not th.any(th.isnan(z))


def test_encoder_width_invariant(device: th.device) -> None:
    # the global pooling makes z defined for any chunk width
    encoder = ChunkEncoder([(2, 4), (4, 8)], [2, 2], 4)
    encoder.to(device)
    encoder.eval()

    assert encoder(th.randn(2, 2, 16, 16, device=device)).size() == (2, 4)
    assert encoder(th.randn(2, 2, 16, 64, device=device)).size() == (2, 4)


def test_encoder_bad_channels() -> None:
    with pytest.raises(AssertionError):
        ChunkEncoder([(2, 4), (8, 16)], [2, 2], 4)

    with pytest.raises(AssertionError):
        ChunkEncoder([(2, 4), (4, 8)], [2], 4)
