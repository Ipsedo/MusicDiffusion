# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join

import torch as th
from tqdm import tqdm

from .data import (
    N_FFT,
    OUTPUT_SIZES,
    SAMPLE_RATE,
    STFT_STRIDE,
    magnitude_phase_to_wav,
)
from .networks import Denoiser
from .options import GenerateOptions, ModelOptions


def generate(
    model_options: ModelOptions, generate_options: GenerateOptions
) -> None:

    if not exists(generate_options.output_dir):
        mkdir(generate_options.output_dir)
    elif not isdir(generate_options.output_dir):
        raise NotADirectoryError(generate_options.output_dir)

    print("Load model...")

    # pylint: disable=duplicate-code
    denoiser = Denoiser(
        model_options.input_channels,
        model_options.steps,
        model_options.beta_1,
        model_options.beta_t,
        model_options.unet_channels,
        model_options.norm_groups,
    )
    # pylint: enable=duplicate-code

    device = "cuda" if model_options.cuda else "cpu"

    denoiser.load_state_dict(
        th.load(generate_options.denoiser_dict_state, map_location=device)
    )

    denoiser.eval()

    if model_options.cuda:
        denoiser.cuda()

    height, width = OUTPUT_SIZES

    with th.no_grad():

        print("Pass rand data to generator...")

        x_t = th.randn(
            generate_options.musics,
            model_options.input_channels,
            height,
            width * generate_options.frames,
            device=device,
        )

        x_0 = (
            denoiser.fast_sample(
                x_t, generate_options.fast_sample, verbose=True
            )
            if generate_options.fast_sample is not None
            else denoiser.sample(x_t, verbose=True)
        )

        print("Saving sound...")

        for i in tqdm(range(x_0.size(0))):
            out_sound_path = join(
                generate_options.output_dir, f"sound_{i}.wav"
            )

            magnitude_phase_to_wav(
                x_0[i, None, :, :, :].detach().cpu(),
                out_sound_path,
                SAMPLE_RATE,
                N_FFT,
                STFT_STRIDE,
            )
