# -*- coding: utf-8 -*-
import re
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
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

        self.__dataset_path = dataset_path

    def __getitem__(self, index: int) -> th.Tensor:
        magn_phase: th.Tensor = th.load(
            join(self.__dataset_path, self.__all_files[index])
        )

        return magn_phase

    def __len__(self) -> int:
        return len(self.__all_files)
