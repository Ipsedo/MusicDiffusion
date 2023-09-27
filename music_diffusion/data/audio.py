# -*- coding: utf-8 -*-
import glob
from os import mkdir
from os.path import exists, isdir, join
from typing import Literal, Tuple

import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f
from tqdm import tqdm

from . import constants
from .primitive import simpson


def diff(x: th.Tensor) -> th.Tensor:
    return th_f.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0), "constant", 0)


def unwrap(phi: th.Tensor) -> th.Tensor:
    d_phi = diff(phi)
    d_phi_m = ((d_phi + th.pi) % (2 * th.pi)) - th.pi
    d_phi_m[(d_phi_m == -th.pi) & (d_phi > 0)] = th.pi
    phi_adj = d_phi_m - d_phi
    phi_adj[d_phi.abs() < th.pi] = 0
    return phi + phi_adj.cumsum(1)


def bark_scale(
    magnitude: th.Tensor, mode: Literal["scale", "unscale"]
) -> th.Tensor:
    assert (
        len(magnitude.size()) == 2
    ), f"(STFT, TIME), actual = {magnitude.size()}"

    min_hz = 20.0
    max_hz = constants.SAMPLE_RATE // 2

    lin_space: th.Tensor = (
        th.linspace(min_hz, max_hz, magnitude.size()[0]) / 600.0
    )
    scale = 6.0 * th.arcsinh(lin_space)[:, None]
    scale = scale / scale[-1, :]

    res: th.Tensor = (
        magnitude / scale if mode == "unscale" else magnitude * scale
    )
    return res


# copied code from
# https://github.com/magenta/magenta/blob/main/magenta/models/gansynth/lib/spectral_ops.py
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def mel_to_hertz(mel_values: th.Tensor) -> th.Tensor:
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        th.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def hertz_to_mel(frequencies_hertz: th.Tensor) -> th.Tensor:
    return _MEL_HIGH_FREQUENCY_Q * th.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def linear_to_mel_weight_matrix(
    num_mel_bins: int = constants.N_FFT // 2,
    num_spectrogram_bins: int = constants.N_FFT // 2,
    sample_rate: int = constants.SAMPLE_RATE,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> th.Tensor:

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = th.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[
        bands_to_zero:, None
    ]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = th.linspace(
        hertz_to_mel(th.tensor(lower_edge_hertz)).item(),
        hertz_to_mel(th.tensor(upper_edge_hertz)).item(),
        num_mel_bins + 2,
    )

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * th.log(rhs + th.sqrt(1.0 + rhs**2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[None, :]
    center_hz = mel_to_hertz(center_mel)[None, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[None, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (
        center_hz - lower_edge_hz
    )
    upper_slopes = (upper_edge_hz - linear_frequencies) / (
        upper_edge_hz - center_hz
    )

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = th.maximum(
        th.tensor(0.0), th.minimum(lower_slopes, upper_slopes)
    )

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = th_f.pad(
        mel_weights_matrix, [bands_to_zero, 0, 0, 0], "constant"
    )
    return mel_weights_matrix


# end of copied code


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

    # magnitude = bark_scale(magnitude, "scale")
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
    # magnitude = bark_scale(magnitude, "unscale")

    phase = (phase + 1.0) / 2.0 * 2.0 * th.pi - th.pi
    phase = simpson(th.zeros(phase.size()[0], 1), phase, 1, 1.0)
    phase = phase % (2 * th.pi)

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

    idx = 0

    for wav_p in tqdm(w_p):
        complex_values = wav_to_stft(
            wav_p, n_per_seg=constants.N_FFT, stride=constants.STFT_STRIDE
        )

        if complex_values.size()[1] < constants.N_VEC:
            continue

        magnitude, phase = stft_to_magnitude_phase(
            complex_values, nb_vec=constants.N_VEC
        )

        nb_sample = magnitude.size()[0]

        for s_idx in range(nb_sample):
            magnitude_phase_path = join(
                dataset_output_dir, f"magn_phase_{idx}.pt"
            )

            magnitude_phase = th.stack(
                [magnitude[s_idx, :, :], phase[s_idx, :, :]], dim=0
            )
            magnitude_phase = magnitude_phase.to(th.float)

            th.save(magnitude_phase, magnitude_phase_path)

            idx += 1
