from os import mkdir
from os.path import exists, isdir, join
from typing import List, NamedTuple, Tuple

import torch as th
from torchvision.transforms import Compose
from tqdm import tqdm

from .data import (
    N_FFT,
    OUTPUT_SIZES,
    SAMPLE_RATE,
    STFT_STRIDE,
    ChannelMinMaxNorm,
    RangeChange,
    magn_phase_to_wav,
)
from .networks import Denoiser

GenerateOptions = NamedTuple(
    "GenerateOptions",
    [
        ("denoiser_dict_state", str),
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("unet_channels", List[Tuple[int, int]]),
        ("use_attentions", List[bool]),
        ("attention_heads", int),
        ("time_size", int),
        ("cuda", bool),
        ("output_dir", str),
        ("frames", int),
        ("musics", int),
    ],
)


def generate(generate_options: GenerateOptions) -> None:

    if not exists(generate_options.output_dir):
        mkdir(generate_options.output_dir)
    elif not isdir(generate_options.output_dir):
        raise NotADirectoryError(generate_options.output_dir)

    print("Load model...")

    denoiser = Denoiser(
        generate_options.input_channels,
        generate_options.steps,
        generate_options.time_size,
        generate_options.beta_1,
        generate_options.beta_t,
        generate_options.unet_channels,
        generate_options.use_attentions,
        generate_options.attention_heads,
    )

    device = "cuda" if generate_options.cuda else "cpu"

    denoiser.load_state_dict(
        th.load(generate_options.denoiser_dict_state, map_location=device)
    )

    denoiser.eval()

    if generate_options.cuda:
        denoiser.cuda()

    transform = Compose(
        [
            ChannelMinMaxNorm(),
            RangeChange(-1.0, 1.0),
        ]
    )

    height, width = OUTPUT_SIZES

    with th.no_grad():

        print("Pass rand data to generator...")

        x_t = th.randn(
            generate_options.musics,
            generate_options.input_channels,
            height,
            width * generate_options.frames,
            device=device,
        )

        x_0 = denoiser.sample(x_t, verbose=True)
        x_0 = transform(x_0)

        print("Saving sound...")

        for i in tqdm(range(x_0.size(0))):
            out_sound_path = join(
                generate_options.output_dir, f"sound_{i}.wav"
            )

            magn_phase_to_wav(
                x_0[i, None, :, :, :].detach().cpu(),
                out_sound_path,
                SAMPLE_RATE,
                N_FFT,
                STFT_STRIDE,
            )
