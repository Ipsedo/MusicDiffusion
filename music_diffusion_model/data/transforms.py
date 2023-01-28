from abc import ABCMeta, abstractmethod

import torch as th


class ImgTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        pass


class ChannelMinMaxNorm(ImgTransform):
    def __call__(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 4

        x_max = th.amax(x, dim=(-2, -1), keepdim=True)
        x_min = th.amin(x, dim=(-2, -1), keepdim=True)

        res: th.Tensor = (x - x_min) / (x_max - x_min)

        return res


class RangeChange(ImgTransform):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        res: th.Tensor = (img_data + self.__lower_bound) * (
            self.__upper_bound - self.__lower_bound
        )
        return res


class ChangeType(ImgTransform):
    def __init__(self, dtype: th.dtype) -> None:
        self.__dtype = dtype

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        return img_data.to(self.__dtype)
