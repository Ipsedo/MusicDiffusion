import pytest
import torch as th

from music_diffusion.networks import drop_condition


def test_drop_condition_zero_probability(device: th.device) -> None:
    z = th.randn(8, 4, device=device)

    assert th.equal(drop_condition(z, 0.0), z)


def test_drop_condition_one_probability(device: th.device) -> None:
    z = th.randn(8, 4, device=device)

    assert th.all(th.eq(drop_condition(z, 1.0), 0.0))


def test_drop_condition_per_sample(device: th.device) -> None:
    th.manual_seed(0)

    z = th.ones(1024, 4, device=device)
    dropped = drop_condition(z, 0.5)

    # a sample is either fully kept or fully dropped
    row_sum = dropped.sum(dim=1)
    assert th.all((row_sum == 0.0) | (row_sum == 4.0))

    ratio = (row_sum == 0.0).float().mean().item()
    assert 0.4 < ratio < 0.6


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_drop_condition_bad_probability(p: float) -> None:
    with pytest.raises(AssertionError):
        drop_condition(th.randn(2, 2), p)
