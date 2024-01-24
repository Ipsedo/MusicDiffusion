# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join

import pandas as pd
import torch as th
from tqdm import tqdm

from .data import (
    N_FFT,
    OUTPUT_SIZES,
    SAMPLE_RATE,
    STFT_STRIDE,
    magnitude_phase_to_wav,
)
from .options import GenerateOptions, ModelOptions


def generate(
    model_options: ModelOptions, generate_options: GenerateOptions
) -> None:

    assert generate_options.musics == len(generate_options.keys)
    assert generate_options.musics == len(generate_options.scoring_list)

    if not exists(generate_options.output_dir):
        mkdir(generate_options.output_dir)
    elif not isdir(generate_options.output_dir):
        raise NotADirectoryError(generate_options.output_dir)

    print("Load model...")

    denoiser = model_options.new_denoiser()

    device = "cuda" if model_options.cuda else "cpu"

    loaded_state_dict = th.load(
        generate_options.denoiser_dict_state, map_location=device
    )

    ema_prefix = "ema_model."

    state_dict = (
        {
            k[len(ema_prefix) :]: p
            for k, p in loaded_state_dict.items()
            if k.startswith(ema_prefix)
        }
        if generate_options.ema_denoiser
        else loaded_state_dict
    )

    denoiser.load_state_dict(state_dict)

    denoiser.eval()

    print(f"Parameters : {denoiser.count_parameters()}")

    if model_options.cuda:
        denoiser.cuda()

    height, width = OUTPUT_SIZES

    with th.no_grad():

        print("Pass rand data to generator...")

        x_t = th.randn(
            generate_options.musics,
            model_options.unet_channels[0][0],
            height,
            width * generate_options.frames,
            device=device,
        )
        y = generate_options.get_y().to(device)

        x_0 = (
            denoiser.fast_sample(
                x_t, y, generate_options.fast_sample, verbose=True
            )
            if generate_options.fast_sample is not None
            else denoiser.sample(x_t, y, verbose=True)
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
                magn_scale=generate_options.magn_scale,
            )

        condition_df = pd.DataFrame(
            [
                [i, generate_options.keys[i], generate_options.scoring_list[i]]
                for i in range(len(generate_options.keys))
            ],
            columns=["id", "key", "scoring"],
        )

        condition_df.to_csv(
            join(generate_options.output_dir, "conditions.csv"), index=False
        )
