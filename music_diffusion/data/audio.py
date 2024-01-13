# -*- coding: utf-8 -*-
import json
from os import mkdir
from os.path import exists, isdir, join
from typing import Literal, Tuple

import pandas as pd
import torch as th
import torch.nn.functional as th_f
import torchaudio as th_audio
import torchaudio.functional as th_audio_f
from tqdm import tqdm

from . import constants
from .metadata import (
    create_genre_to_idx_dict,
    create_key_to_idx_dict,
    create_scoring_to_idx_dict,
    multi_label_one_hot_encode,
    one_hot_encode,
)
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
    threshold: float = 1.0 / 2**8,
    magn_scale: float = 1.0,
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
    magnitude[magnitude < threshold] = 0.0
    magnitude = bark_scale(magnitude, "unscale")
    magnitude = magnitude * magn_scale

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
    metadata_csv_path: str,
    dataset_output_dir: str,
) -> None:

    if not exists(dataset_output_dir):
        mkdir(dataset_output_dir)
    elif not isdir(dataset_output_dir):
        raise NotADirectoryError(dataset_output_dir)

    metadata_df = pd.read_csv(metadata_csv_path, sep=";")
    metadata_df["scoring"] = (
        metadata_df["scoring"]
        .str.replace(r"([\[\]'])", "", regex=True)
        .apply(lambda s: s.split(", "))
    )

    key_to_idx = create_key_to_idx_dict(metadata_df)
    genre_to_idx = create_genre_to_idx_dict(metadata_df)
    scoring_to_idx = create_scoring_to_idx_dict(metadata_df)

    with open(
        join(dataset_output_dir, "key_to_idx.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(key_to_idx, f)

    with open(
        join(dataset_output_dir, "genre_to_idx.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(genre_to_idx, f)

    with open(
        join(dataset_output_dir, "scoring_to_idx.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(scoring_to_idx, f)

    metadata_df["key_ohe"] = metadata_df["key"].apply(
        lambda k: one_hot_encode(k, key_to_idx)
    )
    metadata_df["genre_ohe"] = metadata_df["genre"].apply(
        lambda g: one_hot_encode(g, genre_to_idx)
    )
    metadata_df["scoring_ohe"] = metadata_df["scoring"].apply(
        lambda s: multi_label_one_hot_encode(s, scoring_to_idx)
    )

    idx = 0

    idx_to_bwv = {}

    tqdm_bar = tqdm(metadata_df.iterrows())

    for _, row in tqdm_bar:
        wav_p = row["wav_path"]
        bwv = row["bwv"]

        # save metadata
        key = row["key_ohe"]
        genre = row["genre_ohe"]
        scoring = row["scoring_ohe"]

        th.save(key, join(dataset_output_dir, f"key_{bwv}.pt"))
        th.save(genre, join(dataset_output_dir, f"genre_{bwv}.pt"))
        th.save(scoring, join(dataset_output_dir, f"scoring_{bwv}.pt"))

        # convert audio
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

            idx_to_bwv[idx] = bwv

            idx += 1

        tqdm_bar.set_description(f"total : {idx}")

    # save idx_to_bwv to CSV
    idx_to_bwv_df = pd.DataFrame(
        [[idx, bwv] for idx, bwv in idx_to_bwv.items()], columns=["idx", "bwv"]
    )
    idx_to_bwv_df.to_csv(
        join(dataset_output_dir, "idx_to_bwv.csv"), sep=";", index=False
    )
