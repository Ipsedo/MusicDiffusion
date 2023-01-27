import gzip
import pickle as pkl
from os.path import abspath, dirname, join

import torch as th
from torch.utils.data import Dataset

_RESOURCE_FOLDER = abspath(join(dirname(__file__), "..", "resources"))


class MNISTDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        (mnist_train, _), _ = pkl.load(
            gzip.open(join(_RESOURCE_FOLDER, "mnist.pkl.gz"), "rb"),
            encoding="bytes",
        )

        self.__tensor = th.from_numpy(mnist_train)

    def __getitem__(self, index: int) -> th.Tensor:
        # None -> one channel
        return 2.0 * self.__tensor[index, None] / 255.0 - 1.0

    def __len__(self) -> int:
        return self.__tensor.size(0)
