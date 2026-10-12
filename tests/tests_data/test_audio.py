# pylint: disable=duplicate-code
from os import remove
from os.path import dirname, exists, isfile, join
from pathlib import Path

import numpy as np
import pytest
import torch as th
import torchaudio as th_audio

from music_diffusion.data import (
    MAGN_MAX,
    MAGN_MIN,
    SAMPLE_RATE,
    destandardize_magnitude,
    magnitude_phase_to_wav,
    standardize_magnitude,
    stft_to_magnitude_phase,
    wav_to_stft,
)
from music_diffusion.data.audio import diff, unwrap

# stereo, 44100 Hz, 230132 samples per channel
_EXAMPLE_44100_PATH = join(dirname(__file__), "resources", "example.wav")
_EXAMPLE_44100_LENGTH = 230132
_EXAMPLE_44100_SR = 44100


def _write_sinusoids(
    wav_path: Path, bins: list[int], n_per_seg: int, duration_s: float
) -> None:
    """Write a mono wav at SAMPLE_RATE made of sinusoids whose frequencies
    fall exactly on the given STFT bins (f = k * SAMPLE_RATE / n_per_seg)."""
    t = th.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    wav = th.zeros_like(t)
    for k in bins:
        f0 = k * SAMPLE_RATE / n_per_seg
        wav += th.sin(2.0 * th.pi * f0 * t)
    wav = wav / len(bins)
    th_audio.save(str(wav_path), wav[None, :], SAMPLE_RATE)


# ---------------------------------------------------------------------------
# existing shape tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# diff / unwrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("length", [2, 17, 64])
def test_diff_values(batch_size: int, length: int) -> None:
    x = th.randn(batch_size, length)

    d = diff(x)

    assert d.size() == x.size()
    assert th.equal(d[:, 0], th.zeros(batch_size))
    assert th.equal(d[:, 1:], x[:, 1:] - x[:, :-1])


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("length", [8, 100, 513])
def test_unwrap_matches_numpy(batch_size: int, length: int) -> None:
    # float64 : in float32 the cumsum error grows with the unwrapped
    # value (~1e-4 for |phi| ~ 100) and makes the comparison flaky
    phi = (th.rand(batch_size, length, dtype=th.float64) * 6.0 - 3.0) * th.pi

    unwrapped = unwrap(phi)

    expected = th.from_numpy(np.unwrap(phi.numpy(), axis=1))
    assert unwrapped.size() == phi.size()
    assert th.allclose(unwrapped, expected, atol=1e-9)


@pytest.mark.parametrize("jump_sign", [1.0, -1.0])
@pytest.mark.parametrize("jump_idx", [1, 5, 9])
def test_unwrap_removes_exact_two_pi_jump(
    jump_sign: float, jump_idx: int
) -> None:
    length = 10
    phi = th.zeros(1, length)
    phi[:, jump_idx:] = jump_sign * 2.0 * th.pi

    unwrapped = unwrap(phi)

    assert th.allclose(unwrapped, th.zeros(1, length), atol=1e-5)
    assert th.allclose(diff(unwrapped), th.zeros(1, length), atol=1e-5)


# ---------------------------------------------------------------------------
# wav_to_stft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_per_seg", [256, 512, 1024])
@pytest.mark.parametrize("k", [3, 17, 60])
def test_wav_to_stft_pure_sinusoid_peak_bin(
    tmp_path: Path, n_per_seg: int, k: int
) -> None:
    wav_file = tmp_path / "sin.wav"
    _write_sinusoids(wav_file, [k], n_per_seg, duration_s=1.0)

    stft = wav_to_stft(str(wav_file), n_per_seg, n_per_seg // 4)
    magnitude = stft.abs()

    assert magnitude.size(0) == n_per_seg // 2
    mean_spectrum = magnitude.mean(dim=1)
    assert mean_spectrum.argmax().item() == k
    # the bin is dominant in every interior frame, not only on average
    # (first / last frames are affected by the reflect padding of center=True)
    assert th.all(magnitude[:, 1:-1].argmax(dim=0) == k)


@pytest.mark.parametrize(
    "n_per_seg, stride", [(1024, 128), (512, 64), (256, 256)]
)
def test_wav_to_stft_resample_and_mix(n_per_seg: int, stride: int) -> None:
    raw, sr = th_audio.load(_EXAMPLE_44100_PATH)
    assert raw.size() == (2, _EXAMPLE_44100_LENGTH)
    assert sr == _EXAMPLE_44100_SR

    stft = wav_to_stft(_EXAMPLE_44100_PATH, n_per_seg, stride)

    len_16k = round(_EXAMPLE_44100_LENGTH * SAMPLE_RATE / _EXAMPLE_44100_SR)
    expected_frames = len_16k // stride + 1
    assert stft.size(0) == n_per_seg // 2
    assert abs(stft.size(1) - expected_frames) <= 1
    assert th.is_complex(stft)
    assert not th.isnan(stft).any()


# ---------------------------------------------------------------------------
# stft_to_magnitude_phase
# ---------------------------------------------------------------------------


def _stft_with_shifted_peaks(nfft: int, stft_nb: int) -> th.Tensor:
    """Column j has its magnitude maximum at bin j % nfft."""
    magnitude = th.full((nfft, stft_nb), 0.1)
    for j in range(stft_nb):
        magnitude[j % nfft, j] = 1.0
    return th.complex(magnitude, th.zeros_like(magnitude))


@pytest.mark.parametrize("nfft", [8, 16])
@pytest.mark.parametrize("nb_vec", [4, 8])
@pytest.mark.parametrize("q", [1, 3])
@pytest.mark.parametrize("r", [1, 2])
def test_stft_to_magn_phase_drops_leading_frames(
    nfft: int, nb_vec: int, q: int, r: int
) -> None:
    stft_nb = q * nb_vec + r
    stft = _stft_with_shifted_peaks(nfft, stft_nb)

    magn, phase = stft_to_magnitude_phase(stft, nb_vec)

    assert magn.size() == (q, nfft, nb_vec)
    assert phase.size() == (q, nfft, nb_vec)

    # first observation == STFT frames r : r + nb_vec (leading frames dropped)
    expected_bins = th.tensor([(r + i) % nfft for i in range(nb_vec)])
    assert th.equal(magn[0].argmax(dim=0), expected_bins)
    # last observation ends on the last STFT frame
    expected_last = th.tensor(
        [(stft_nb - nb_vec + i) % nfft for i in range(nb_vec)]
    )
    assert th.equal(magn[-1].argmax(dim=0), expected_last)


@pytest.mark.parametrize("nfft", [8, 16])
@pytest.mark.parametrize("nb_vec", [4, 8])
@pytest.mark.parametrize("q", [1, 2])
def test_stft_to_magn_phase_remainder_nb_vec_minus_one(
    nfft: int, nb_vec: int, q: int
) -> None:
    # Edge case: the left zero pad adds one frame, so the truncation is
    # computed on stft_nb + 1 frames. When r == nb_vec - 1, nothing is
    # dropped and the first observation starts with the pad frame.
    r = nb_vec - 1
    stft_nb = q * nb_vec + r
    stft = _stft_with_shifted_peaks(nfft, stft_nb)

    magn, _ = stft_to_magnitude_phase(stft, nb_vec)

    assert magn.size() == (q + 1, nfft, nb_vec)
    assert th.equal(magn[0, :, 0], th.full((nfft,), -1.0))
    expected_bins = th.tensor([i % nfft for i in range(nb_vec - 1)])
    assert th.equal(magn[0, :, 1:].argmax(dim=0), expected_bins)


@pytest.mark.parametrize("nfft", [8, 32])
@pytest.mark.parametrize("top_db", [60.0, 80.0])
def test_stft_to_magn_phase_dynamic_range(nfft: int, top_db: float) -> None:
    stft_nb = 6
    magnitude = th.ones(nfft, stft_nb)
    # column 2 : exactly at the floor, column 4 : below the floor
    magnitude[:, 2] = 10.0 ** (-top_db / 20.0)
    magnitude[:, 4] = 10.0 ** (-(top_db + 20.0) / 20.0)
    stft = th.complex(magnitude, th.zeros_like(magnitude))

    # nb_vec = stft_nb + 1 keeps every frame (incl. the left pad) in one obs
    magn, _ = stft_to_magnitude_phase(stft, nb_vec=stft_nb + 1, top_db=top_db)

    assert magn.size() == (1, nfft, stft_nb + 1)
    # left zero pad -> -inf dB clamped to -top_db -> exactly -1
    assert th.equal(magn[0, :, 0], th.full((nfft,), -1.0))
    # below the floor -> clamped -> exactly -1 (STFT column 4 -> index 5)
    assert th.equal(magn[0, :, 5], th.full((nfft,), -1.0))
    # at the floor -> ~ -1 (epsilon inside the log keeps it just above)
    assert th.allclose(magn[0, :, 3], th.full((nfft,), -1.0), atol=1e-4)
    # maximum -> ~ +1
    assert th.allclose(magn[0, :, 1], th.full((nfft,), 1.0), atol=1e-4)
    assert th.allclose(magn[0, :, 2], th.full((nfft,), 1.0), atol=1e-4)


@pytest.mark.parametrize("nfft", [8, 32])
@pytest.mark.parametrize("stft_nb", [5, 15])
def test_stft_to_magn_phase_constant_magnitude(
    nfft: int, stft_nb: int
) -> None:
    magnitude = th.full((nfft, stft_nb), 0.37)
    stft = th.complex(magnitude, th.zeros_like(magnitude))

    magn, _ = stft_to_magnitude_phase(stft, nb_vec=stft_nb + 1)

    assert magn.size() == (1, nfft, stft_nb + 1)
    assert th.equal(magn[0, :, 0], th.full((nfft,), -1.0))
    assert th.allclose(magn[0, :, 1:], th.ones(nfft, stft_nb), atol=1e-4)


@pytest.mark.parametrize("nfft", [8, 64])
@pytest.mark.parametrize("stft_nb, nb_vec", [(10, 16), (1, 3), (30, 32)])
def test_stft_to_magn_phase_too_short(
    nfft: int, stft_nb: int, nb_vec: int
) -> None:
    # Documents the current (silent) behaviour: when there are fewer frames
    # than nb_vec - 1 (the left pad adds one frame), the result is an empty
    # observation of shape (1, F, 0).
    # create_dataset filters this case upstream (stft frames < N_VEC).
    stft = th.complex(th.randn(nfft, stft_nb), th.randn(nfft, stft_nb))

    magn, phase = stft_to_magnitude_phase(stft, nb_vec)

    assert magn.size() == (1, nfft, 0)
    assert phase.size() == (1, nfft, 0)


@pytest.mark.parametrize("nfft", [8, 64])
@pytest.mark.parametrize("nb_vec", [2, 16])
def test_stft_to_magn_phase_nb_vec_minus_one_frames(
    nfft: int, nb_vec: int
) -> None:
    # Edge case of the above: stft_nb == nb_vec - 1 frames still yield one
    # full observation because the left zero pad completes it.
    stft_nb = nb_vec - 1
    stft = th.complex(th.randn(nfft, stft_nb), th.randn(nfft, stft_nb))

    magn, phase = stft_to_magnitude_phase(stft, nb_vec)

    assert magn.size() == (1, nfft, nb_vec)
    assert phase.size() == (1, nfft, nb_vec)
    assert th.equal(magn[0, :, 0], th.full((nfft,), -1.0))


# ---------------------------------------------------------------------------
# standardize_magnitude / destandardize_magnitude
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", [(16, 32), (512, 512)])
def test_standardize_magnitude_3d(sizes: tuple[int, int]) -> None:
    magn_phase = th.rand(2, *sizes) * 2.0 - 1.0

    standardized = standardize_magnitude(magn_phase)

    assert standardized.size() == magn_phase.size()
    assert th.equal(standardized[1], magn_phase[1])
    assert th.allclose(
        destandardize_magnitude(standardized), magn_phase, atol=1e-6
    )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batched", [True, False])
def test_standardize_magnitude_wrong_channels(
    channels: int, batched: bool
) -> None:
    sizes = (channels, 8, 8)
    magn_phase = th.rand(2, *sizes) if batched else th.rand(*sizes)

    with pytest.raises(AssertionError):
        standardize_magnitude(magn_phase)

    with pytest.raises(AssertionError):
        destandardize_magnitude(magn_phase)


def test_standardize_magnitude_bounds() -> None:
    magnitude = th.tensor([[-1.0, 1.0], [1.0, -1.0]])
    phase = th.tensor([[0.25, -0.5], [0.0, 1.0]])
    magn_phase = th.stack([magnitude, phase], dim=0)

    standardized = standardize_magnitude(magn_phase)

    expected = th.tensor([[MAGN_MIN, MAGN_MAX], [MAGN_MAX, MAGN_MIN]])
    assert th.allclose(standardized[0], expected, atol=1e-6)
    assert th.equal(standardized[1], phase)

    destandardized = destandardize_magnitude(
        th.stack([expected, phase], dim=0)
    )
    assert th.allclose(destandardized[0], magnitude, atol=1e-6)
    assert th.equal(destandardized[1], phase)


# ---------------------------------------------------------------------------
# magnitude_phase_to_wav
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("nfft, stride", [(128, 32), (256, 64)])
@pytest.mark.parametrize("nb_vec", [16, 32])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_magn_phase_to_wav_properties(
    tmp_path: Path,
    batch_size: int,
    nfft: int,
    stride: int,
    nb_vec: int,
    sample_rate: int,
) -> None:
    wav_file = tmp_path / "out.wav"
    magn_phase = th.randn(batch_size, 2, nfft // 2, nb_vec)

    magnitude_phase_to_wav(
        magn_phase, str(wav_file), sample_rate, nfft, stride
    )
    wav, sr = th_audio.load(str(wav_file))

    assert sr == sample_rate
    assert wav.size(0) == 1
    # inverse_spectrogram with center=True (default) and length=None
    # yields (frames - 1) * hop samples, where frames = batch_size * nb_vec
    assert wav.size(1) == (batch_size * nb_vec - 1) * stride
    # peak normalised to 1 before saving (16-bit quantisation tolerance)
    assert abs(wav.abs().max().item() - 1.0) < 1e-2
    assert not th.isnan(wav).any()


@pytest.mark.parametrize("nfft", [128, 512])
@pytest.mark.parametrize("nb_vec", [16, 64])
def test_magn_phase_to_wav_silence(
    tmp_path: Path, nfft: int, nb_vec: int
) -> None:
    wav_file = tmp_path / "silence.wav"
    magnitude = th.full((1, nfft // 2, nb_vec), -1.0)
    phase = th.zeros(1, nfft // 2, nb_vec)
    magn_phase = th.stack([magnitude, phase], dim=1)

    magnitude_phase_to_wav(
        magn_phase, str(wav_file), SAMPLE_RATE, nfft, nfft // 4
    )
    wav, _ = th_audio.load(str(wav_file))

    assert not th.isnan(wav).any()
    assert not th.isinf(wav).any()


@pytest.mark.parametrize("nfft", [128, 256])
def test_magn_phase_to_wav_wrong_channels(tmp_path: Path, nfft: int) -> None:
    magn_phase = th.randn(1, 3, nfft // 2, 16)

    with pytest.raises(AssertionError, match="Channels"):
        magnitude_phase_to_wav(
            magn_phase, str(tmp_path / "out.wav"), SAMPLE_RATE, nfft
        )


@pytest.mark.parametrize("nfft", [128, 256])
@pytest.mark.parametrize("height_offset", [-1, 1, 64])
def test_magn_phase_to_wav_wrong_frequency(
    tmp_path: Path, nfft: int, height_offset: int
) -> None:
    magn_phase = th.randn(1, 2, nfft // 2 + height_offset, 16)

    with pytest.raises(AssertionError, match="Frequency"):
        magnitude_phase_to_wav(
            magn_phase, str(tmp_path / "out.wav"), SAMPLE_RATE, nfft
        )


@pytest.mark.parametrize("ndim", [2, 3, 5])
def test_magn_phase_to_wav_wrong_ndim(tmp_path: Path, ndim: int) -> None:
    nfft = 128
    sizes = [2, nfft // 2, 16, 1, 1][:ndim]
    magn_phase = th.randn(*sizes)

    with pytest.raises(AssertionError):
        magnitude_phase_to_wav(
            magn_phase, str(tmp_path / "out.wav"), SAMPLE_RATE, nfft
        )


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_fft, stride, nb_vec, bins",
    [
        (256, 64, 128, [5, 20]),
        (512, 128, 64, [10, 40]),
        (1024, 128, 32, [3, 100]),
    ],
)
def test_round_trip_keeps_dominant_bins(
    tmp_path: Path,
    n_fft: int,
    stride: int,
    nb_vec: int,
    bins: list[int],
) -> None:
    in_file = tmp_path / "in.wav"
    out_file = tmp_path / "out.wav"
    _write_sinusoids(in_file, bins, n_fft, duration_s=2.0)

    stft = wav_to_stft(str(in_file), n_fft, stride)
    original_top = stft.abs().mean(dim=1).topk(2).indices.sort().values
    assert original_top.tolist() == sorted(bins)

    magn, phase = stft_to_magnitude_phase(stft, nb_vec)
    assert magn.size(0) >= 1
    magn_phase = th.stack([magn, phase], dim=1)
    assert magn_phase.size() == (magn.size(0), 2, n_fft // 2, nb_vec)

    magnitude_phase_to_wav(
        magn_phase, str(out_file), SAMPLE_RATE, n_fft, stride
    )

    # phase is only approximately reconstructed : compare spectral peaks
    stft_back = wav_to_stft(str(out_file), n_fft, stride)
    back_top = stft_back.abs().mean(dim=1).topk(2).indices.sort().values
    assert th.equal(back_top, original_top)
