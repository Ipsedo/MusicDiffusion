import gzip
import pickle as pkl
import re
from os import listdir
from os.path import abspath, dirname, isdir, isfile, join

import numpy as np
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

_RESOURCE_FOLDER = abspath(join(dirname(__file__), "..", "resources"))


class MNISTDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        (mnist_train, _), (mnist_valid, _) = pkl.load(
            gzip.open(join(_RESOURCE_FOLDER, "mnist.pkl.gz"), "rb"),
            encoding="bytes",
        )

        mnist = np.concatenate([mnist_train, mnist_valid], axis=0)

        self.__tensor = th.from_numpy(mnist)

    def __getitem__(self, index: int) -> th.Tensor:
        return self.__tensor[index][None]

    def __len__(self) -> int:
        return self.__tensor.size(0)


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
