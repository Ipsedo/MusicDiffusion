from typing import Callable, Tuple

import torch as th
import torch.nn as nn


class TimeEmbeder(nn.Embedding):
    def __init__(self, steps: int, size: int):
        super().__init__(steps, size)


class TimeWrapper(nn.Module):
    def __init__(
        self,
        module: Callable[[th.Tensor], th.Tensor],
        steps: int,
        time_size: int,
    ):
        super().__init__()

        self.__emb = TimeEmbeder(steps, time_size)

        self.__module = module

    def forward(self, x_and_time: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        x, t = x_and_time
        assert len(x.size()) == 5
        assert len(t.size()) == 1

        b, _, _, w, h = x.size()

        time_vec = self.__emb(t)
        time_vec = time_vec[None, :, :, None, None].repeat(b, 1, 1, w, h)
        x_time = th.cat([x, time_vec], dim=2)

        return self.__module(x_time)
