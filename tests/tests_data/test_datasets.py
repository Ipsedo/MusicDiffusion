# pylint: disable=duplicate-code
import re
from os import listdir
from pathlib import Path

import pytest
import torch as th
import torchaudio as th_audio
from torch.utils.data import DataLoader

from music_diffusion.data import (
    N_FFT,
    N_VEC,
    SAMPLE_RATE,
    AudioDataset,
    create_dataset,
    stft_to_magnitude_phase,
    wav_to_stft,
)
from music_diffusion.data.constants import MAGN_MEAN, MAGN_STD

_FILE_RE = re.compile(r"^magn_phase_(\d+)_(\d+)\.pt$")


def _save_wav(path: Path, seconds: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    gen = th.Generator().manual_seed(seed)
    nb_samples = int(seconds * SAMPLE_RATE)
    t = th.arange(nb_samples, dtype=th.float32) / SAMPLE_RATE

    # sine (440 Hz) + noise, stereo, in [-1, 1]
    sine = th.sin(2.0 * th.pi * 440.0 * t)
    noise = th.rand(nb_samples, generator=gen) * 2.0 - 1.0
    wav = 0.5 * sine + 0.25 * noise

    th_audio.save(str(path), th.stack([wav, wav]), SAMPLE_RATE)


def _expected_nb_samples(wav: Path) -> int:
    complex_values = wav_to_stft(str(wav))

    if complex_values.size()[1] < N_VEC:
        return 0

    magnitude, _ = stft_to_magnitude_phase(complex_values)

    return int(magnitude.size()[0])


def _dataset_entries(dataset_dir: Path) -> list[tuple[int, int]]:
    entries = []

    for f in listdir(dataset_dir):
        match = _FILE_RE.match(f)
        assert match is not None, f"unexpected file {f}"
        entries.append((int(match.group(1)), int(match.group(2))))

    return sorted(entries)


def _expected_entries(nb_chunks: list[int]) -> list[tuple[int, int]]:
    return [
        (song, chunk)
        for song, nb in enumerate(nb_chunks)
        for chunk in range(nb)
    ]


@pytest.fixture(name="audio_dir")
def get_audio_dir(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "audio"

    _save_wav(audio_dir / "long.wav", 10.0, seed=0)
    _save_wav(audio_dir / "short.wav", 0.5, seed=1)

    return audio_dir


@pytest.fixture(name="dataset_dir")
def get_dataset_dir(tmp_path: Path, audio_dir: Path) -> Path:
    dataset_dir = tmp_path / "out"

    create_dataset(str(audio_dir / "*.wav"), str(dataset_dir))

    return dataset_dir


# create_dataset


def test_create_dataset_files(audio_dir: Path, dataset_dir: Path) -> None:
    nb_long = _expected_nb_samples(audio_dir / "long.wav")
    nb_short = _expected_nb_samples(audio_dir / "short.wav")

    assert nb_long >= 2
    assert nb_short == 0

    assert dataset_dir.is_dir()
    # the short file produces no chunk and takes no song index
    assert _dataset_entries(dataset_dir) == _expected_entries([nb_long])


def test_create_dataset_content(dataset_dir: Path) -> None:
    files = listdir(dataset_dir)

    assert len(files) > 0

    for f in files:
        magn_phase: th.Tensor = th.load(dataset_dir / f)

        assert magn_phase.dtype == th.float32
        assert magn_phase.size() == (2, N_FFT // 2, N_VEC)
        assert not th.any(th.isnan(magn_phase))
        assert th.all(th.ge(magn_phase[0], -1.0))
        assert th.all(th.le(magn_phase[0], 1.0))
        assert th.all(th.ge(magn_phase[1], -1.0))
        assert th.all(th.le(magn_phase[1], 1.0))


@pytest.mark.parametrize("seconds", [(10.0, 15.0), (15.0, 10.0)])
def test_create_dataset_song_index(
    tmp_path: Path, seconds: tuple[float, float]
) -> None:
    audio_dir = tmp_path / "audio"
    wav_a = audio_dir / "a.wav"
    wav_b = audio_dir / "b.wav"

    _save_wav(wav_a, seconds[0], seed=0)
    _save_wav(wav_b, seconds[1], seed=1)

    nb_a = _expected_nb_samples(wav_a)
    nb_b = _expected_nb_samples(wav_b)

    assert nb_a >= 2
    assert nb_b >= 2

    out_dir = tmp_path / "out"
    create_dataset(str(audio_dir / "*.wav"), str(out_dir))

    entries = _dataset_entries(out_dir)

    # two songs, each with its own contiguous chunk indices
    songs = sorted({song for song, _ in entries})
    assert songs == [0, 1]

    nb_per_song = sorted(
        len([c for s, c in entries if s == song]) for song in songs
    )
    assert nb_per_song == sorted([nb_a, nb_b])

    for song in songs:
        chunks = sorted(c for s, c in entries if s == song)
        assert chunks == list(range(len(chunks)))


def test_create_dataset_creates_output_dir(
    tmp_path: Path, audio_dir: Path
) -> None:
    out_dir = tmp_path / "new_out"

    assert not out_dir.exists()

    create_dataset(str(audio_dir / "long.wav"), str(out_dir))

    assert out_dir.is_dir()
    assert len(listdir(out_dir)) == _expected_nb_samples(
        audio_dir / "long.wav"
    )


def test_create_dataset_reuses_existing_dir(
    tmp_path: Path, audio_dir: Path
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    foreign = out_dir / "foreign.txt"
    foreign.write_text("keep me")

    create_dataset(str(audio_dir / "long.wav"), str(out_dir))

    assert foreign.is_file()
    assert foreign.read_text() == "keep me"

    pt_files = [f for f in listdir(out_dir) if _FILE_RE.match(f)]
    assert len(pt_files) == _expected_nb_samples(audio_dir / "long.wav")
    assert set(listdir(out_dir)) == set(pt_files) | {"foreign.txt"}


def test_create_dataset_output_is_file(
    tmp_path: Path, audio_dir: Path
) -> None:
    out_file = tmp_path / "out"
    out_file.write_text("not a directory")

    with pytest.raises(NotADirectoryError):
        create_dataset(str(audio_dir / "long.wav"), str(out_file))


def test_create_dataset_empty_glob(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    create_dataset(str(tmp_path / "nothing" / "*.wav"), str(out_dir))

    assert out_dir.is_dir()
    assert listdir(out_dir) == []


def test_create_dataset_recursive_glob(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    nested_wav = audio_dir / "sub" / "deeper" / "long.wav"

    _save_wav(nested_wav, 10.0, seed=0)

    out_dir = tmp_path / "out"
    create_dataset(str(audio_dir / "**" / "*.wav"), str(out_dir))

    assert _dataset_entries(out_dir) == _expected_entries(
        [_expected_nb_samples(nested_wav)]
    )


# AudioDataset


def _make_item(
    song: int, chunk: int, sizes: tuple[int, int] = (8, 8)
) -> th.Tensor:
    # magnitude channel filled with the song, phase channel with the chunk
    magn = th.full(sizes, float(song))
    phase = th.full(sizes, float(chunk))

    return th.stack([magn, phase], dim=0)


def _save_items(
    dataset_dir: Path, nb_chunks: list[int], sizes: tuple[int, int] = (8, 8)
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for song, nb in enumerate(nb_chunks):
        for chunk in range(nb):
            th.save(
                _make_item(song, chunk, sizes),
                dataset_dir / f"magn_phase_{song}_{chunk}.pt",
            )


def _song_chunk(item: th.Tensor) -> tuple[int, int]:
    # undo the magnitude standardization
    song = item[0, 0, 0].item() * MAGN_STD + MAGN_MEAN
    chunk = item[1, 0, 0].item()

    return round(song), round(chunk)


@pytest.fixture(name="mixed_dir")
def get_mixed_dir(tmp_path: Path) -> Path:
    mixed_dir = tmp_path / "mixed"
    _save_items(mixed_dir, [2, 1])

    # ignored entries
    th.save(_make_item(9, 9), mixed_dir / "other.pt")
    th.save(_make_item(9, 9), mixed_dir / "magn_phase_x_0.pt")
    th.save(_make_item(9, 9), mixed_dir / "magn_phase_0_1.pt.bak")
    (mixed_dir / "magn_phase_5_0.pt").mkdir()

    return mixed_dir


def test_audio_dataset_filters_files(mixed_dir: Path) -> None:
    dataset = AudioDataset(str(mixed_dir))

    assert len(dataset) == 3


def test_audio_dataset_numeric_order(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _save_items(dataset_dir, [11, 2])

    dataset = AudioDataset(str(dataset_dir))

    # sorted by (song, chunk) as integers, not lexicographically
    expected = [(0, c) for c in range(11)] + [(1, 0), (1, 1)]

    for pos, (song, chunk) in enumerate(expected):
        assert dataset.song_of(pos) == song
        _, x_target = dataset[pos]
        assert _song_chunk(x_target) == (song, chunk)


@pytest.mark.parametrize("sizes", [(8, 8), (16, 32)])
def test_audio_dataset_getitem(tmp_path: Path, sizes: tuple[int, int]) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    saved = th.stack(
        [th.rand(*sizes) * 2.0 - 1.0, th.rand(*sizes) * 2.0 - 1.0], dim=0
    )
    th.save(saved, dataset_dir / "magn_phase_0_0.pt")

    dataset = AudioDataset(str(dataset_dir))
    x_ref, x_target = dataset[0]

    assert x_target.size() == (2, *sizes)
    assert th.equal(x_target[1], saved[1])
    assert th.allclose(x_target[0], (saved[0] - MAGN_MEAN) / MAGN_STD)

    # single chunk song : the reference is the target itself
    assert th.equal(x_ref, x_target)


def test_audio_dataset_pair_same_song(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _save_items(dataset_dir, [4, 1, 3])

    dataset = AudioDataset(str(dataset_dir))

    th.manual_seed(0)

    for pos in range(len(dataset)):  # pylint: disable=consider-using-enumerate
        for _ in range(8):
            x_ref, x_target = dataset[pos]

            ref_song, ref_chunk = _song_chunk(x_ref)
            target_song, target_chunk = _song_chunk(x_target)

            assert ref_song == target_song == dataset.song_of(pos)

            if ref_song == 1:
                assert ref_chunk == target_chunk == 0
            else:
                assert ref_chunk != target_chunk


def test_audio_dataset_reference_covers_song(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _save_items(dataset_dir, [5])

    dataset = AudioDataset(str(dataset_dir))

    th.manual_seed(0)

    # the reference of chunk 0 is drawn among every other chunk of the song
    seen = {dataset.reference_position(0) for _ in range(256)}

    assert seen == {1, 2, 3, 4}


def test_audio_dataset_legacy_format(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "legacy"
    dataset_dir.mkdir()

    th.save(_make_item(0, 0), dataset_dir / "magn_phase_0.pt")

    with pytest.raises(RuntimeError, match="create_data"):
        AudioDataset(str(dataset_dir))


def test_audio_dataset_not_a_dir(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("")

    with pytest.raises(AssertionError):
        AudioDataset(str(not_a_dir))

    with pytest.raises(AssertionError):
        AudioDataset(str(tmp_path / "missing"))


def test_audio_dataset_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    dataset = AudioDataset(str(empty_dir))

    assert len(dataset) == 0


def test_audio_dataset_integration(dataset_dir: Path) -> None:
    nb_files = len(listdir(dataset_dir))

    dataset = AudioDataset(str(dataset_dir))

    assert len(dataset) == nb_files
    assert nb_files >= 2

    x_ref, x_target = dataset[0]
    assert x_target.size() == (2, N_FFT // 2, N_VEC)
    assert x_target.dtype == th.float32
    assert x_ref.size() == x_target.size()
    assert not th.equal(x_ref, x_target)

    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch_ref, batch_target = next(iter(loader))

    assert batch_ref.size() == (2, 2, N_FFT // 2, N_VEC)
    assert batch_target.size() == (2, 2, N_FFT // 2, N_VEC)
