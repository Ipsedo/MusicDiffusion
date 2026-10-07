from os import mkdir
from os.path import exists, isdir, join

import matplotlib.pyplot as plt
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
    destandardize_magnitude,
    magnitude_phase_to_wav,
)
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
        x_ref: th.Tensor,
    ) -> None:
        """x_ref : (nb_sample, C, H, W) standardized reference chunks, one
        identity per periodic sample."""

        if not exists(output_dir):
            mkdir(output_dir)
        elif not isdir(output_dir):
            raise NotADirectoryError(output_dir)

        assert len(x_ref.size()) == 4
        assert x_ref.size(1) == in_channels

        self.__output_dir = output_dir
        self.__save_every = save_every
        self.__nb_sample = x_ref.size(0)
        self.__x_ref = x_ref

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

        # save string repr
        with open(
            join(self.__output_dir, "noiser.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.__noiser))

        with open(
            join(self.__output_dir, "denoiser.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.__denoiser))

        # save the reference chunks once, to compare with the samples
        x_ref_cpu = destandardize_magnitude(self.__x_ref.detach().cpu())
        th.save(x_ref_cpu, join(self.__output_dir, "reference_magn_phase.pt"))

        for i in range(self.__nb_sample):
            magnitude_phase_to_wav(
                x_ref_cpu[i, None],
                join(self.__output_dir, f"reference_ID{i}.wav"),
                SAMPLE_RATE,
                N_FFT,
                STFT_STRIDE,
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

                self.__ema_denoiser.eval()
                ema_denoiser = self.__ema_denoiser.ema_model
                z = ema_denoiser.encode(self.__x_ref.to(device))
                x_0 = ema_denoiser.sample(x_t, z, verbose=True)
                self.__ema_denoiser.train()

                x_0 = destandardize_magnitude(x_0)

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
