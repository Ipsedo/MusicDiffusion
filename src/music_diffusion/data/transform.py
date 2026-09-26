from abc import ABC, abstractmethod

import torch as th

# pylint: disable=too-few-public-methods


class ImgTransform(ABC):
    @abstractmethod
    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        pass


class ChannelMinMaxNorm(ImgTransform):
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.__epsilon = epsilon

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        assert len(img_data.size()) == 4

        x_max = th.amax(img_data, dim=(-2, -1), keepdim=True)
        x_min = th.amin(img_data, dim=(-2, -1), keepdim=True)

        out: th.Tensor = (img_data - x_min) / (x_max - x_min + self.__epsilon)

        return out


class InverseRangeChange(ImgTransform):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        return (img_data - self.__lower_bound) / (
            self.__upper_bound - self.__lower_bound
        )


class RangeChange(ImgTransform):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        out: th.Tensor = (
            img_data * (self.__upper_bound - self.__lower_bound)
            + self.__lower_bound
        )
        return out


class ChangeType(ImgTransform):
    def __init__(self, dtype: th.dtype) -> None:
        self.__dtype = dtype

    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        return img_data.to(self.__dtype)


# pylint: enable=too-few-public-methods
