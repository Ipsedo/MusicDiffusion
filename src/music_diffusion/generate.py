from os import mkdir
from os.path import exists, isdir, join

import torch as th
from tqdm import tqdm

from .data import (
    N_FFT,
    OUTPUT_SIZES,
    SAMPLE_RATE,
    STFT_STRIDE,
    destandardize_magnitude,
    magnitude_phase_to_wav,
)
from .options import GenerateOptions, ModelOptions


def generate(
    model_options: ModelOptions, generate_options: GenerateOptions
) -> None:

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

    def __sample(curr_x_t: th.Tensor, curr_z: th.Tensor) -> th.Tensor:
        return (
            denoiser.fast_sample(
                curr_x_t, generate_options.fast_sample, curr_z, verbose=True
            )
            if generate_options.fast_sample is not None
            else denoiser.sample(curr_x_t, curr_z, verbose=True)
        )

    with th.no_grad():

        # 1. one frame per music without condition
        print("Generate first frame without condition...")

        x_first = __sample(
            th.randn(
                generate_options.musics,
                model_options.unet_channels[0][0],
                height,
                width,
                device=device,
            ),
            denoiser.null_condition(generate_options.musics, device),
        )

        # 2. the identity of each music is the one of its first frame
        print("Encode first frame...")

        z = denoiser.encode(x_first)

        # 3. the whole music, conditioned by this identity
        print(f"Generate {generate_options.frames} frames with condition...")

        x_t = th.randn(
            generate_options.musics,
            model_options.unet_channels[0][0],
            height,
            width * generate_options.frames,
            device=device,
        )

        x_0 = __sample(x_t, z)

        x_0 = destandardize_magnitude(x_0)

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
