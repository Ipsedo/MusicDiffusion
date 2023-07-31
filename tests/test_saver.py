# -*- coding: utf-8 -*-
from os import mkdir
from os.path import dirname, exists, isfile, join
from shutil import rmtree

import pytest
import torch as th

from music_diffusion.networks import Denoiser, Noiser
from music_diffusion.saver import Saver


@pytest.mark.parametrize("save_every", [2, 3])
@pytest.mark.parametrize("nb_samples", [2, 3])
def test_saver(save_every: int, nb_samples: int) -> None:
    noiser = Noiser(10, 1e-4, 2e-2)
    denoiser = Denoiser(2, 3, 1e-4, 2e-2, [(8, 16)], 8)
    optim = th.optim.Adam(denoiser.parameters())

    tmp_dir = join(dirname(__file__), "__tmp_dir__")
    mkdir(tmp_dir)

    saver = Saver(2, noiser, denoiser, optim, tmp_dir, save_every, nb_samples)
    try:
        for _ in range(save_every - 1):
            saver.save()

            assert not exists(join(tmp_dir, "denoiser_0.pt"))
            assert not exists(join(tmp_dir, "denoiser_optim_0.pt"))
            assert not exists(join(tmp_dir, "noiser_0.pt"))

            for i in range(nb_samples):
                assert not exists(join(tmp_dir, f"magn_phase_0_ID{i}.png"))
                assert not exists(join(tmp_dir, f"sample_0_ID{i}.wav"))

        saver.save()

        assert exists(join(tmp_dir, "denoiser_0.pt")) and isfile(
            join(tmp_dir, "denoiser_0.pt")
        )
        assert exists(join(tmp_dir, "denoiser_optim_0.pt")) and isfile(
            join(tmp_dir, "denoiser_optim_0.pt")
        )
        assert exists(join(tmp_dir, "noiser_0.pt")) and isfile(
            join(tmp_dir, "noiser_0.pt")
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
