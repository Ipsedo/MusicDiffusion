import glob
from os import mkdir
from os.path import exists, isdir, join
from typing import Literal, Tuple

import numpy as np
import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f
from tqdm import tqdm

from . import constants


def diff(x: th.Tensor) -> th.Tensor:
    return th_f.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0), "constant", 0)


def unwrap(phi: th.Tensor) -> th.Tensor:
    d_phi = diff(phi)
    d_phi_m = ((d_phi + np.pi) % (2 * np.pi)) - np.pi
    d_phi_m[(d_phi_m == -np.pi) & (d_phi > 0)] = np.pi
    phi_adj = d_phi_m - d_phi
    phi_adj[d_phi.abs() < np.pi] = 0
    return phi + phi_adj.cumsum(1)


def bark_scale(
    magnitude: th.Tensor, mode: Literal["scale", "unscale"]
) -> th.Tensor:
    assert (
        len(magnitude.size()) == 2
    ), f"(STFT, TIME), actual = {magnitude.size()}"

    min_hz = 20.0
    max_hz = 44100 // 2

    lin_space: th.Tensor = (
        th.linspace(min_hz, max_hz, magnitude.size()[0]) / 600.0
    )
    scale = 6.0 * th.arcsinh(lin_space)[:, None]
    scale = scale / scale[-1, :]

    res: th.Tensor = (
        magnitude / scale if mode == "unscale" else magnitude * scale
    )
    return res


def simpson(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    sizes = derivative.size()
    n = derivative.size()[dim]

    evens = th.arange(0, n, 2)
    odds = th.arange(1, n, 2)

    even_derivative = th.index_select(derivative, dim, evens)
    odd_derivative = th.index_select(derivative, dim, odds)

    shift_odd_derivative = th_f.pad(
        odd_derivative,
        [
            p
            for d in reversed(range(len(sizes)))
            for p in [1 if d == dim else 0, 0]
        ],
        "constant",
        0,
    )

    even_primitive = first_primitive + dx / 3 * (
        (
            2 * even_derivative
            + 4
            * th.index_select(
                shift_odd_derivative,
                dim=dim,
                index=th.arange(0, even_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - th.select(even_derivative, dim, 0).unsqueeze(dim)
        - th.select(even_derivative, dim, 0).unsqueeze(dim)
    )

    odd_primitive = (dx / 3) * (
        (
            2 * odd_derivative
            + 4
            * th.index_select(
                even_derivative,
                dim=dim,
                index=th.arange(0, odd_derivative.size()[dim]),
            )
        ).cumsum(dim)
        - 4 * th.select(even_derivative, dim, 0).unsqueeze(dim)
        - th.select(odd_derivative, dim, 0).unsqueeze(dim)
        - odd_derivative
    )

    odd_primitive += first_primitive + dx / 12 * (
        5 * th.select(derivative, dim, 0)
        + 8 * th.select(derivative, dim, 1)
        - th.select(derivative, dim, 2)
    ).unsqueeze(dim)

    primitive = th.zeros_like(derivative)

    view = [-1 if i == dim else 1 for i in range(len(sizes))]
    repeat = [1 if i == dim else s for i, s in enumerate(sizes)]
    evens = evens.view(*view).repeat(*repeat)
    odds = odds.view(*view).repeat(*repeat)

    primitive.scatter_(dim, evens, even_primitive)
    primitive.scatter_(dim, odds, odd_primitive)

    return primitive


def trapezoid(
    first_primitive: th.Tensor,
    derivative: th.Tensor,
    dim: int,
    dx: float,
) -> th.Tensor:
    return first_primitive + dx * (
        derivative.cumsum(dim=dim)
        - derivative / 2.0
        - th.select(derivative, dim, 0).unsqueeze(dim) / 2.0
    )


def wav_to_stft(
    wav_p: str,
    n_per_seg: int = constants.N_FFT,
    stride: int = constants.STFT_STRIDE,
    epsilon: float = 1e-8,
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)

    assert sr == constants.SAMPLE_RATE, (
        f"Audio sample rate must be {constants.SAMPLE_RATE}Hz, "
        f'file "{wav_p}" is {sr}Hz'
    )

    raw_audio_mono = raw_audio.mean(0)
    raw_audio_mono = (
        2
        * (raw_audio_mono - raw_audio_mono.min())
        / (raw_audio_mono.max() - raw_audio_mono.min() + epsilon)
        - 1.0
    )

    assert -1.0 <= raw_audio_mono.min() <= 1.0
    assert -1.0 <= raw_audio_mono.max() <= 1.0

    complex_values: th.Tensor = th_audio_f.spectrogram(
        raw_audio_mono,
        pad=0,
        window=th.hann_window(n_per_seg),
        n_fft=n_per_seg,
        hop_length=stride,
        win_length=n_per_seg,
        power=None,
        normalized=True,
    )

    # remove Nyquist frequency
    return complex_values[:-1, :]


def stft_to_magnitude_phase(
    complex_values: th.Tensor,
    nb_vec: int = constants.N_VEC,
    epsilon: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor]:
    magnitude = th.abs(complex_values)
    phase = th.angle(complex_values)

    magnitude = bark_scale(magnitude, "scale")
    magnitude = th_f.pad(magnitude, (1, 0, 0, 0), "constant", 0.0)

    phase = unwrap(phase)
    phase = th_f.pad(phase, (1, 0, 0, 0), "constant", 0.0)
    phase = th.gradient(phase, dim=1, spacing=1.0, edge_order=1)[0]

    max_magnitude = magnitude.max()
    min_magnitude = magnitude.min()
    magnitude = (
        2
        * (magnitude - min_magnitude)
        / (max_magnitude - min_magnitude + epsilon)
        - 1
    )

    max_phase = phase.max()
    min_phase = phase.min()
    phase = 2 * (phase - min_phase) / (max_phase - min_phase + epsilon) - 1

    magnitude = magnitude[:, magnitude.size()[1] % nb_vec :]
    phase = phase[:, phase.size()[1] % nb_vec :]
    magnitude = th.stack(magnitude.split(nb_vec, dim=1), dim=0)
    phase = th.stack(phase.split(nb_vec, dim=1), dim=0)

    return magnitude, phase


def magnitude_phase_to_wav(
    magnitude_phase: th.Tensor,
    wav_path: str,
    sample_rate: int,
    n_fft: int = constants.N_FFT,
    stft_stride: int = constants.STFT_STRIDE,
) -> None:
    assert (
        len(magnitude_phase.size()) == 4
    ), f"(N, 2, H, W), actual = {magnitude_phase.size()}"

    assert (
        magnitude_phase.size()[1] == 2
    ), f"Channels must be equal to 2, actual = {magnitude_phase.size()[1]}"

    assert magnitude_phase.size()[2] == n_fft // 2, (
        f"Frequency size must be equal to {n_fft // 2}, "
        f"actual = {magnitude_phase.size()[2]}"
    )

    magnitude_phase_flattened = magnitude_phase.permute(1, 2, 0, 3).flatten(
        2, 3
    )
    magnitude = magnitude_phase_flattened[0, :, :]
    phase = magnitude_phase_flattened[1, :, :]

    magnitude = (magnitude + 1.0) / 2.0
    magnitude = bark_scale(magnitude, "unscale")

    phase = (phase + 1.0) / 2.0 * 2.0 * np.pi - np.pi
    phase = simpson(th.zeros(phase.size()[0], 1), phase, 1, 1.0)
    phase = phase % (2 * np.pi)

    real = magnitude * th.cos(phase)
    imaginary = magnitude * th.sin(phase)

    real_res = th_f.pad(real, (0, 0, 0, 1), "constant", 0)
    imaginary_res = th_f.pad(imaginary, (0, 0, 0, 1), "constant", 0)

    z = real_res + imaginary_res * 1j

    raw_audio = th_audio_f.inverse_spectrogram(
        z,
        length=None,
        pad=0,
        window=th.hann_window(n_fft),
        n_fft=n_fft,
        hop_length=stft_stride,
        win_length=n_fft,
        normalized=True,
    )

    th_audio.save(wav_path, raw_audio[None, :], sample_rate)


def create_dataset(
    audio_path: str,
    dataset_output_dir: str,
) -> None:

    w_p = glob.glob(audio_path)

    if not exists(dataset_output_dir):
        mkdir(dataset_output_dir)
    elif not isdir(dataset_output_dir):
        raise NotADirectoryError(dataset_output_dir)

    n_per_seg = constants.N_FFT
    stride = constants.STFT_STRIDE

    nb_vec = constants.N_VEC

    idx = 0

    for wav_p in tqdm(w_p):
        complex_values = wav_to_stft(wav_p, n_per_seg=n_per_seg, stride=stride)

        if complex_values.size()[1] < nb_vec:
            continue

        magnitude, phase = stft_to_magnitude_phase(
            complex_values, nb_vec=nb_vec
        )

        nb_sample = magnitude.size()[0]

        for s_idx in range(nb_sample):
            s_magnitude = magnitude[s_idx, :, :].to(th.float64)
            s_phase = phase[s_idx, :, :].to(th.float64)

            magnitude_phase_path = join(
                dataset_output_dir, f"magn_phase_{idx}.pt"
            )

            magnitude_phase = th.stack([s_magnitude, s_phase], dim=0)

            th.save(magnitude_phase, magnitude_phase_path)

            idx += 1
