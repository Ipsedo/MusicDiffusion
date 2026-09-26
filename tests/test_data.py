from os import remove
from os.path import exists, isfile

import pytest
import torch as th

from music_diffusion.data import (
    MAGN_MAX,
    MAGN_MIN,
    destandardize_magnitude,
    magnitude_phase_to_wav,
    simpson,
    standardize_magnitude,
    stft_to_magnitude_phase,
    trapezoid,
    wav_to_stft,
)


@pytest.mark.parametrize("dx", [0.01, 0.02, 0.04])
def test_simpson(dx: float) -> None:
    start = -8.0
    end = 8.0

    steps = int((end - start) / dx)

    delta = 1e-2
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = simpson(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)


@pytest.mark.parametrize("dx", [0.01, 0.02, 0.04])
def test_trapezoid(dx: float) -> None:
    start = -8.0
    end = 8.0

    steps = int((end - start) / dx)

    delta = 1e-2
    dim = 1

    derivative = th.cos(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )
    primitive = th.sin(th.linspace(start, end, steps))[None, :, None].repeat(
        20, 1, 10
    )

    res_simpson = trapezoid(
        th.select(primitive, dim, 0).unsqueeze(dim), derivative, dim, dx
    )

    assert th.all(th.abs(primitive - res_simpson).mean(dim=dim) < delta)


@pytest.mark.parametrize("nperseg", [256, 512, 1024])
@pytest.mark.parametrize("stride", [64, 128, 256])
def test_wav_to_stft(wav_path: str, nperseg: int, stride: int) -> None:
    stft = wav_to_stft(wav_path, nperseg, stride)

    assert len(stft.size()) == 2
    assert stft.size()[0] == nperseg // 2
    assert th.is_complex(stft)


@pytest.mark.parametrize("nfft", [128, 256, 512])
@pytest.mark.parametrize("stft_nb", [1024, 2048, 4096])
@pytest.mark.parametrize("nb_vec", [128, 256, 512])
def test_stft_to_magn_phase(nfft: int, stft_nb: int, nb_vec: int) -> None:
    size = (nfft, stft_nb)
    nb_observation = stft_nb // nb_vec

    stft = th.complex(th.randn(*size), th.randn(*size))
    magn, phase = stft_to_magnitude_phase(stft, nb_vec, epsilon=1e-8)

    assert magn.size() == (nb_observation, nfft, nb_vec)
    assert th.all(th.logical_and(th.ge(magn, -1), th.le(magn, 1)))

    assert phase.size() == (nb_observation, nfft, nb_vec)
    assert th.all(th.logical_and(th.ge(phase, -1), th.le(phase, 1)))


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("nfft", [128, 256, 512])
@pytest.mark.parametrize("nb_vec", [128, 256, 512])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100])
def test_magn_phase_to_wav(
    batch_size: int, nfft: int, nb_vec: int, sample_rate: int
) -> None:
    wav_path = "./tmp.wav"

    try:
        magn_phase = th.randn(batch_size, 2, nfft // 2, nb_vec)

        magnitude_phase_to_wav(
            magn_phase, wav_path, sample_rate, nfft, nfft // 2
        )

        assert exists(wav_path)
        assert isfile(wav_path)
    finally:
        if exists(wav_path):
            remove(wav_path)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sizes", [(16, 32), (32, 32)])
def test_standardize_magnitude(
    batch_size: int, sizes: tuple[int, int]
) -> None:
    magn_phase = th.rand(batch_size, 2, *sizes) * 2.0 - 1.0

    standardized = standardize_magnitude(magn_phase)

    assert standardized.size() == magn_phase.size()
    assert th.all(th.ge(standardized[:, 0], MAGN_MIN - 1e-5))
    assert th.all(th.le(standardized[:, 0], MAGN_MAX + 1e-5))
    assert th.equal(standardized[:, 1], magn_phase[:, 1])

    assert th.allclose(
        destandardize_magnitude(standardized), magn_phase, atol=1e-6
    )
