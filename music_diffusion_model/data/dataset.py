import gzip
import pickle as pkl
from os.path import abspath, dirname, join

import numpy as np
import torch as th
from torch.utils.data import Dataset

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
