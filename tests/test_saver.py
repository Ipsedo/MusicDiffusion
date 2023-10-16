# -*- coding: utf-8 -*-
from os import mkdir
from os.path import dirname, exists, isfile, join
from shutil import rmtree

import pytest
import torch as th
from ema_pytorch import EMA

from music_diffusion.networks import Denoiser, Noiser
from music_diffusion.saver import Saver


@pytest.mark.parametrize("save_every", [2, 3])
@pytest.mark.parametrize("nb_samples", [2, 3])
def test_saver(save_every: int, nb_samples: int) -> None:
    steps = 1
    channels = 2

    noiser = Noiser(steps)
    denoiser = Denoiser(channels, steps, 1, [(2, 4)], 1)
    optim = th.optim.Adam(denoiser.parameters())
    ema = EMA(denoiser)

    tmp_dir = join(dirname(__file__), "__tmp_dir__")
    mkdir(tmp_dir)

    saver = Saver(
        channels, noiser, denoiser, optim, ema, tmp_dir, save_every, nb_samples
    )
    try:
        for _ in range(save_every - 1):
            saver.save()

            assert not exists(join(tmp_dir, "denoiser_0.pt"))
            assert not exists(join(tmp_dir, "denoiser_ema_0.pt"))
            assert not exists(join(tmp_dir, "denoiser_optim_0.pt"))
            assert not exists(join(tmp_dir, "noiser_0.pt"))
            assert not exists(join(tmp_dir, "magn_phase_0.pt"))

            for i in range(nb_samples):
                assert not exists(join(tmp_dir, f"magn_phase_0_ID{i}.png"))
                assert not exists(join(tmp_dir, f"sample_0_ID{i}.wav"))

        saver.save()

        assert exists(join(tmp_dir, "denoiser_0.pt")) and isfile(
            join(tmp_dir, "denoiser_0.pt")
        )
        assert exists(join(tmp_dir, "denoiser_ema_0.pt")) and isfile(
            join(tmp_dir, "denoiser_ema_0.pt")
        )
        assert exists(join(tmp_dir, "denoiser_optim_0.pt")) and isfile(
            join(tmp_dir, "denoiser_optim_0.pt")
        )
        assert exists(join(tmp_dir, "noiser_0.pt")) and isfile(
            join(tmp_dir, "noiser_0.pt")
        )
        assert exists(join(tmp_dir, "magn_phase_0.pt")) and isfile(
            join(tmp_dir, "magn_phase_0.pt")
        )

        for i in range(nb_samples):
            assert exists(join(tmp_dir, f"magn_phase_0_ID{i}.png")) and isfile(
                join(tmp_dir, f"magn_phase_0_ID{i}.png")
            )
            assert exists(join(tmp_dir, f"sample_0_ID{i}.wav")) and isfile(
                join(tmp_dir, f"sample_0_ID{i}.wav")
            )

    finally:
        rmtree(tmp_dir)
