# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join

import torch as th
from ema_pytorch import EMA
from torch.optim.optimizer import Optimizer

from .data import N_SAMPLES, SAMPLE_RATE, tensor_to_wav
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
        nb_audio: int,
        nb_sample: int = N_SAMPLES,
    ) -> None:

        if not exists(output_dir):
            mkdir(output_dir)
        elif not isdir(output_dir):
            raise NotADirectoryError(output_dir)

        self.__output_dir = output_dir
        self.__save_every = save_every
        self.__nb_audio = nb_audio
        self.__nb_sample = nb_sample

        self.__in_channels = in_channels
        self.__noiser = noiser
        self.__denoiser = denoiser
        self.__denoiser_optim = denoiser_optim
        self.__ema_denoiser = ema_denoiser

        self.__curr_save = -1
        self.__curr_idx = 0

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
                    self.__nb_audio,
                    self.__in_channels,
                    self.__nb_sample,
                    device=device,
                )

                self.__ema_denoiser.eval()
                x_0 = self.__ema_denoiser.ema_model.sample(x_t, verbose=True)
                self.__ema_denoiser.train()

                th.save(
                    x_0,
                    join(
                        self.__output_dir, f"raw_audio_{self.__curr_save}.pt"
                    ),
                )

                for i in range(self.__nb_audio):
                    tensor_to_wav(
                        join(
                            self.__output_dir,
                            f"sample_{self.__curr_save}_ID{i}.wav",
                        ),
                        x_0[i].clone(),
                        SAMPLE_RATE,
                    )

        self.__curr_idx += 1

    @property
    def curr_save(self) -> int:
        return self.__curr_save

    @property
    def curr_step(self) -> int:
        return self.__curr_idx % self.__save_every
