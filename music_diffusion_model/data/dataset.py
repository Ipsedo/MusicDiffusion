import gzip
import pickle as pkl
from os.path import abspath, dirname, join

import numpy as np
import torch as th
import torch.nn.functional as th_f
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
        # None -> one channel
        # pad 28 * 28 -> 32 * 32
        return (
            th_f.pad(
                self.__tensor[index],
                (2, 2, 2, 2),
                mode="constant",
                value=0.0,
            )[None]
            / 255.0
        )

    def __len__(self) -> int:
        return self.__tensor.size(0)
