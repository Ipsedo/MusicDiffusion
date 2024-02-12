# -*- coding: utf-8 -*-
import re
from os import listdir
from os.path import isdir, isfile, join
from typing import Tuple

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        assert isdir(dataset_path)

        re_files = re.compile(r"^magn_phase_\d+\.pt$")

        all_files = [
            f
            for f in tqdm(listdir(dataset_path))
            if isfile(join(dataset_path, f)) and re_files.match(f)
        ]

        # Avoid data copy on each worker ? => as numpy array
        self.__all_files = np.array(sorted(all_files))

        self._dataset_path = dataset_path

    def __getitem__(self, index: int) -> th.Tensor:
        magn_phase: th.Tensor = th.load(
            join(self._dataset_path, self.__all_files[index])
        )

        return magn_phase

    def __len__(self) -> int:
        return len(self.__all_files)


class ConditionAudioDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        self.__dataset_path = dataset_path

        self.__audio_dataset = AudioDataset(dataset_path)

        idx_to_bwv_df = pd.read_csv(
            join(dataset_path, "idx_to_bwv.csv"), sep=";"
        )
        self.__idx_to_bwv = {
            row["idx"]: row["bwv"] for _, row in idx_to_bwv_df.iterrows()
        }

    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        magn_phase = self.__audio_dataset[index]

        key = th.load(
            join(self.__dataset_path, f"key_{self.__idx_to_bwv[index]}.pt")
        )
        scoring = th.load(
            join(self.__dataset_path, f"scoring_{self.__idx_to_bwv[index]}.pt")
        )

        # y = th.cat([key, scoring], dim=-1)

        return magn_phase, key, scoring

    def __len__(self) -> int:
        return len(self.__idx_to_bwv)
