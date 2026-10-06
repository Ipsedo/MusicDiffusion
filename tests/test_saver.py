from os.path import exists, isfile
from pathlib import Path

import pytest
import torch as th
from ema_pytorch import EMA

from music_diffusion.networks import Denoiser, Noiser
from music_diffusion.saver import Saver


@pytest.mark.parametrize("save_every", [2, 3])
@pytest.mark.parametrize("nb_samples", [2, 3])
def test_saver(tmp_path: Path, save_every: int, nb_samples: int) -> None:
    steps = 2
    channels = 2

    noiser = Noiser(steps)
    denoiser = Denoiser(steps, 1, [(channels, 4)], [1])
    optim = th.optim.Adam(denoiser.parameters())
    ema = EMA(denoiser)

    saver = Saver(
        channels,
        noiser,
        denoiser,
        optim,
        ema,
        str(tmp_path),
        save_every,
        nb_samples,
    )

    for _ in range(save_every - 1):
        saver.save()

        assert not exists(tmp_path / "denoiser_0.pt")
        assert not exists(tmp_path / "denoiser_ema_0.pt")
        assert not exists(tmp_path / "denoiser_optim_0.pt")
        assert not exists(tmp_path / "noiser_0.pt")
        assert not exists(tmp_path / "magn_phase_0.pt")

        for i in range(nb_samples):
            assert not exists(tmp_path / f"magn_phase_0_ID{i}.png")
            assert not exists(tmp_path / f"sample_0_ID{i}.wav")

    saver.save()

    assert exists(tmp_path / "denoiser_0.pt") and isfile(
        tmp_path / "denoiser_0.pt"
    )
    assert exists(tmp_path / "denoiser_ema_0.pt") and isfile(
        tmp_path / "denoiser_ema_0.pt"
    )
    assert exists(tmp_path / "denoiser_optim_0.pt") and isfile(
        tmp_path / "denoiser_optim_0.pt"
    )
    assert exists(tmp_path / "noiser_0.pt") and isfile(
        tmp_path / "noiser_0.pt"
    )
    assert exists(tmp_path / "magn_phase_0.pt") and isfile(
        tmp_path / "magn_phase_0.pt"
    )

    for i in range(nb_samples):
        assert exists(tmp_path / f"magn_phase_0_ID{i}.png") and isfile(
            tmp_path / f"magn_phase_0_ID{i}.png"
        )
        assert exists(tmp_path / f"sample_0_ID{i}.wav") and isfile(
            tmp_path / f"sample_0_ID{i}.wav"
        )
