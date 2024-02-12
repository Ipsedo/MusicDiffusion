# -*- coding: utf-8 -*-
import json
from os import mkdir
from os.path import exists, isdir, join
from random import choices, randint, sample

import matplotlib.pyplot as plt
import pandas as pd
import torch as th
from ema_pytorch import EMA
from torch.optim.optimizer import Optimizer
from torchvision.transforms import Compose

from .data import (
    N_FFT,
    OUTPUT_SIZES,
    SAMPLE_RATE,
    STFT_STRIDE,
    ChangeType,
    InverseRangeChange,
    RangeChange,
    magnitude_phase_to_wav,
)
from .data.metadata import multi_label_one_hot_encode, one_hot_encode
from .networks import Denoiser, Noiser


class Saver:
    def __init__(
        self,
        in_channels: int,
        noiser: Noiser,
        denoiser: Denoiser,
        denoiser_optim: Optimizer,
        ema_denoiser: EMA,
        output_dir: str,
        save_every: int,
        nb_sample: int,
        dataset_path: str,
    ) -> None:

        if not exists(output_dir):
            mkdir(output_dir)
        elif not isdir(output_dir):
            raise NotADirectoryError(output_dir)

        self.__output_dir = output_dir
        self.__save_every = save_every
        self.__nb_sample = nb_sample

        self.__in_channels = in_channels
        self.__noiser = noiser
        self.__denoiser = denoiser
        self.__denoiser_optim = denoiser_optim
        self.__ema_denoiser = ema_denoiser

        self.__curr_save = -1
        self.__curr_idx = 0

        self.__sample_transform = Compose(
            [
                InverseRangeChange(-1, 1),
                RangeChange(0.0, 255.0),
                ChangeType(th.uint8),
            ]
        )

        with open(
            join(dataset_path, "key_to_idx.json"), "r", encoding="utf-8"
        ) as f:
            self.__key_to_idx = json.load(f)
        with open(
            join(dataset_path, "scoring_to_idx.json"), "r", encoding="utf-8"
        ) as f:
            self.__scoring_to_idx = json.load(f)

    def save(self) -> None:
        if self.__curr_idx % self.__save_every == self.__save_every - 1:

            self.__curr_save += 1

            th.save(
                self.__noiser.state_dict(),
                join(self.__output_dir, f"noiser_{self.__curr_save}.pt"),
            )
            th.save(
                self.__denoiser.state_dict(),
                join(self.__output_dir, f"denoiser_{self.__curr_save}.pt"),
            )
            th.save(
                self.__denoiser_optim.state_dict(),
                join(
                    self.__output_dir, f"denoiser_optim_{self.__curr_save}.pt"
                ),
            )
            th.save(
                self.__ema_denoiser.state_dict(),
                join(self.__output_dir, f"denoiser_ema_{self.__curr_save}.pt"),
            )

            with th.no_grad():
                device = (
                    "cuda"
                    if next(self.__denoiser.parameters()).is_cuda
                    else "cpu"
                )

                x_t = th.randn(
                    self.__nb_sample,
                    self.__in_channels,
                    *OUTPUT_SIZES,
                    device=device,
                )

                # random condition
                key = choices(
                    list(self.__key_to_idx.keys()), k=self.__nb_sample
                )
                scoring = [
                    sample(
                        list(self.__scoring_to_idx.keys()),
                        k=randint(1, len(self.__scoring_to_idx)),
                    )
                    for _ in range(self.__nb_sample)
                ]

                condition_df = pd.DataFrame(
                    [[i, k, s] for i, (k, s) in enumerate(zip(key, scoring))],
                    columns=["id", "key", "scoring"],
                )
                condition_df.to_csv(
                    join(
                        self.__output_dir, f"condition_{self.__curr_save}.csv"
                    ),
                    sep=";",
                    index=False,
                )

                key_tensor = th.stack(
                    [one_hot_encode(k, self.__key_to_idx) for k in key], dim=0
                )
                scoring_tensor = th.stack(
                    [
                        multi_label_one_hot_encode(s, self.__scoring_to_idx)
                        for s in scoring
                    ]
                )

                # generate

                self.__ema_denoiser.eval()
                x_0 = self.__ema_denoiser.ema_model.sample(
                    x_t, key_tensor, scoring_tensor, verbose=True
                )
                self.__ema_denoiser.train()

                th.save(
                    x_0,
                    join(
                        self.__output_dir, f"magn_phase_{self.__curr_save}.pt"
                    ),
                )

                for i in range(self.__nb_sample):
                    magn_phase = x_0[i, None].detach().cpu()

                    magn_phase_vizu = self.__sample_transform(magn_phase)[0]
                    magn = magn_phase_vizu[0]
                    phase = magn_phase_vizu[1]

                    # create two subplots
                    fig, (magn_ax, phase_ax) = plt.subplots(1, 2)

                    # Plot magnitude
                    magn_ax.matshow(magn, cmap="plasma")

                    magn_ax.set_title(
                        f"Magnitude, save {self.__curr_save}, sample {i}"
                    )

                    # Plot phase
                    phase_ax.matshow(phase, cmap="plasma")

                    phase_ax.set_title(
                        f"Phase, save {self.__curr_save}, sample {i}"
                    )

                    fig.savefig(
                        join(
                            self.__output_dir,
                            f"magn_phase_{self.__curr_save}_ID{i}.png",
                        )
                    )

                    plt.close()

                    # Save sample to wav
                    magnitude_phase_to_wav(
                        magn_phase,
                        join(
                            self.__output_dir,
                            f"sample_{self.__curr_save}_ID{i}.wav",
                        ),
                        SAMPLE_RATE,
                        N_FFT,
                        STFT_STRIDE,
                    )

        self.__curr_idx += 1

    @property
    def curr_save(self) -> int:
        return self.__curr_save

    @property
    def curr_step(self) -> int:
        return self.__curr_idx % self.__save_every
