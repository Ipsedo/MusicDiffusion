import re
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio import standardize_magnitude


class AudioDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        assert isdir(dataset_path)

        file_regex = re.compile(r"^magn_phase_(\d+)_(\d+)\.pt$")

        entries = [
            (int(m.group(1)), int(m.group(2)), f)
            for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f))
            and (m := file_regex.match(f)) is not None
        ]

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
