# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join
from typing import List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import torch as th
from torch.optim.optimizer import Optimizer
from torchvision.transforms import Compose

from .data import OUTPUT_SIZES, ChangeType, ChannelMinMaxNorm, RangeChange
from .networks import Denoiser, Noiser

ModelOptions = NamedTuple(
    "ModelOptions",
    [
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("norm_groups", int),
        ("unet_channels", List[Tuple[int, int]]),
        ("use_attentions", List[bool]),
        ("attention_heads", int),
        ("time_size", int),
        ("cuda", bool),
    ],
)


class Saver:
    def __init__(
        self,
        channels: int,
        noiser: Noiser,
        denoiser: Denoiser,
        denoiser_optim: Optimizer,
        output_dir: str,
        save_every: int,
        nb_sample: int,
    ) -> None:

        if not exists(output_dir):
            mkdir(output_dir)
        elif not isdir(output_dir):
            raise NotADirectoryError(output_dir)

        self.__output_dir = output_dir
        self.__save_every = save_every
        self.__nb_sample = nb_sample

        self.__channels = channels
        self.__noiser = noiser
        self.__denoiser = denoiser
        self.__denoiser_optim = denoiser_optim

        self.__curr_save = -1
        self.__curr_idx = 0

        self.__sample_transform = Compose(
            [
                ChannelMinMaxNorm(),
                RangeChange(0.0, 255.0),
                ChangeType(th.uint8),
            ]
        )

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

            with th.no_grad():
                device = (
                    "cuda"
                    if next(self.__denoiser.parameters()).is_cuda
                    else "cpu"
                )

                x_t = th.randn(
                    self.__nb_sample,
                    self.__channels,
                    *OUTPUT_SIZES,
                    device=device,
                )

                self.__denoiser.eval()
                x_0 = self.__denoiser.sample(x_t, verbose=True)
                self.__denoiser.train()

                x_0 = self.__sample_transform(x_0)

                for i in range(self.__nb_sample):
                    magn = x_0[i, 0, :, :].detach().cpu().numpy()
                    phase = x_0[i, 1, :, :].detach().cpu().numpy()

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

        self.__curr_idx += 1

    @property
    def curr_save(self) -> int:
        return self.__curr_save

    @property
    def curr_step(self) -> int:
        return self.__curr_idx % self.__save_every
