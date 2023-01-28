from os.path import join

import matplotlib.pyplot as plt
import torch as th
from torch.optim.optimizer import Optimizer

from .networks import Denoiser, Noiser


class Saver:
    def __init__(
        self,
        noiser: Noiser,
        denoiser: Denoiser,
        denoiser_optim: Optimizer,
        output_dir: str,
        save_every: int,
        nb_sample: int,
    ) -> None:

        self.__output_dir = output_dir
        self.__save_every = save_every
        self.__nb_sample = nb_sample

        self.__noiser = noiser
        self.__denoiser = denoiser
        self.__denoiser_optim = denoiser_optim

        self.__curr_save = 0
        self.__curr_idx = 0

    def save(self) -> None:
        if self.__curr_idx % self.__save_every == 0:

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

            # TODO generic
            x_t = th.randn(self.__nb_sample, 1, 32, 32)
            x_0 = self.__denoiser.sample(x_t)

            for i in range(self.__nb_sample):
                fig = plt.figure()
                plt.matshow(x_0[i, 0], cmap="Greys")
                plt.title(f"Save {self.__curr_save}, sample {i}")
                plt.savefig(
                    join(
                        self.__output_dir, f"sample_{self.__curr_save}_{i}.png"
                    )
                )
                plt.close(fig)

            self.__curr_save += 1

        self.__curr_idx += 1
