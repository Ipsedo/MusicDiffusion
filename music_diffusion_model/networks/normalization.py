import torch as th
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super(PixelNorm, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x / th.sqrt(
            x.pow(2.0).mean(dim=1, keepdim=True) + self.__epsilon
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()
