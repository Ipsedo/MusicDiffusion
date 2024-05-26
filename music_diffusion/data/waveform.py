# -*- coding: utf-8 -*-
import glob
from os import mkdir
from os.path import exists, isdir, join
from typing import List

import torch as th
import torchaudio as th_audio
import torchaudio.functional as th_audio_f
from tqdm import tqdm

from . import constants


def wav_to_tensor(
    wav_p: str, wanted_sr: int, epsilon: float = 1e-8
) -> th.Tensor:
    raw_audio, sr = th_audio.load(wav_p)
    raw_audio = th_audio_f.functional.resample(raw_audio, sr, wanted_sr).to(
        th.float
    )

    if len(raw_audio.size()) == 1:
        raw_audio = raw_audio.unsqueeze(0)
    if raw_audio.size(0) == 1:
        raw_audio = raw_audio.repeat(2, 1)

    assert raw_audio.size(0) == 2, f'Needs stereo : "{wav_p}"'

    audio_min, audio_max = th.min(raw_audio), th.max(raw_audio)

    raw_audio_normalized: th.Tensor = (raw_audio - audio_min) / (
        audio_max - audio_min + epsilon
    )

    return raw_audio_normalized


def split_raw_audio(
    raw_audio: th.Tensor,
    n_samples: int,
) -> List[th.Tensor]:
    return [
        t.clone()
        for t in th.split(
            raw_audio,
            n_samples,
            dim=1,
        )
        if t.size(1) == n_samples
    ]


def tensor_to_wav(
    wav_path: str,
    raw_audio: th.Tensor,
    sample_rate: int = constants.SAMPLE_RATE,
) -> None:
    assert len(raw_audio.size()) == 2

    th_audio.save(wav_path, raw_audio, sample_rate)


def create_waveform_dataset(
    audio_path: str,
    dataset_output_dir: str,
) -> None:
    # pylint: disable=duplicate-code
    w_p = glob.glob(audio_path, recursive=True)

    if not exists(dataset_output_dir):
        mkdir(dataset_output_dir)
    elif not isdir(dataset_output_dir):
        raise NotADirectoryError(dataset_output_dir)

    idx = 0
    tqdm_bar = tqdm(w_p)

    for wav_p in tqdm_bar:
        raw_audio = split_raw_audio(
            wav_to_tensor(wav_p, constants.SAMPLE_RATE), constants.N_SAMPLES
        )

        for a in raw_audio:
            raw_audio_path = join(dataset_output_dir, f"raw_audio_{idx}.pt")

            th.save(a.to(th.float), raw_audio_path)

            idx += 1

        tqdm_bar.set_description(f"total : {idx}")
