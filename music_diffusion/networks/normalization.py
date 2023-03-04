import torch as th
from torch import nn


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x / th.sqrt(
            x.pow(2.0).mean(dim=1, keepdim=True) + self.__epsilon
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class LayerNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True)

        return (x - mean) / th.sqrt(var + self.__epsilon)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()
