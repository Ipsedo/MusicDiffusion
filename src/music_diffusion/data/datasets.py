import re
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio import standardize_magnitude


class AudioDataset(Dataset):
    """Dataset of (reference, target) chunk pairs taken from the same song.

    Each item ``i`` returns ``(x_ref, x_target)`` where ``x_target`` is the
    ``i``-th chunk and ``x_ref`` is another chunk of the same song, drawn
    uniformly among the song chunks. The reference is only meant to be seen
    by the conditioning encoder, so the conditioning vector can only carry
    what is shared across the whole piece. A song with a single chunk returns
    the same chunk twice.
    """

    _FILE_RE = re.compile(r"^magn_phase_(\d+)_(\d+)\.pt$")
    _LEGACY_FILE_RE = re.compile(r"^magn_phase_\d+\.pt$")

    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        assert isdir(dataset_path)

        entries = [
            (int(m.group(1)), int(m.group(2)), f)
            for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f))
            and (m := self._FILE_RE.match(f)) is not None
        ]

        if len(entries) == 0 and any(
            self._LEGACY_FILE_RE.match(f) for f in listdir(dataset_path)
        ):
            raise RuntimeError(
                f"'{dataset_path}' only contains legacy 'magn_phase_N.pt' "
                "files without song identity, regenerate it with the "
                "'create_data' command"
            )

        entries.sort()

        # Avoid data copy on each worker ? => as numpy array
        self.__all_files = np.array([f for _, _, f in entries])
        self.__songs = np.array([s for s, _, _ in entries], dtype=np.int64)

        # song id -> positions (in __all_files) of its chunks
        song_positions: dict[int, list[int]] = {}
        for pos, (song, _, _) in enumerate(entries):
            song_positions.setdefault(song, []).append(pos)

        self.__song_positions = {
            song: np.array(positions, dtype=np.int64)
            for song, positions in song_positions.items()
        }

        self.__dataset_path = dataset_path

    def __load(self, position: int) -> th.Tensor:
        magn_phase: th.Tensor = th.load(
            join(self.__dataset_path, self.__all_files[position])
        )

        return standardize_magnitude(magn_phase)

    def reference_position(self, index: int) -> int:
        positions = self.__song_positions[int(self.__songs[index])]

        if len(positions) == 1:
            return index

        others = positions[positions != index]
        choice = int(th.randint(0, len(others), (1,)).item())

        return int(others[choice])

    def song_of(self, index: int) -> int:
        return int(self.__songs[index])

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        x_target = self.__load(index)
        x_ref = self.__load(self.reference_position(index))

        return x_ref, x_target

    def __len__(self) -> int:
        return len(self.__all_files)
