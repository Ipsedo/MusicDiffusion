import math

import torch as th
from torch import nn


class SinusoidTimeEmbedding(nn.Module):
    def __init__(self, steps: int, size: int) -> None:
        super().__init__()

        pos_emb = th.zeros(steps, size)
        position = th.arange(0, steps).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, size, 2, dtype=th.float)
            * th.tensor(-math.log(10000.0) / size)
        )
        pos_emb[:, 0::2] = th.sin(position.float() * div_term)
        pos_emb[:, 1::2] = th.cos(position.float() * div_term)

        self._pos_emb: th.Tensor

        self.register_buffer("_pos_emb", pos_emb)

    def forward(self, t_index: th.Tensor) -> th.Tensor:
        b, t = t_index.size()

        out = th.index_select(self._pos_emb, dim=0, index=t_index.flatten())
        out = th.unflatten(out, 0, (b, t))

        return out


class TimeEmbedding(nn.Module):
    def __init__(self, steps: int, size: int):
        super().__init__()

        self.__emb = nn.Embedding(steps, size)

    def forward(self, t_index: th.Tensor) -> th.Tensor:
        b, t = t_index.size()

        t_index = t_index.flatten()

        out: th.Tensor = self.__emb(t_index)
        out = th.unflatten(out, 0, (b, t))

        return out


class TimeBypass(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.__module = module

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, t = x.size()[:2]

        x = x.flatten(0, 1)
        out: th.Tensor = self.__module(x)
        out = th.unflatten(out, 0, (b, t))

        return out


class TimeToScaleShift(nn.Module):
    def __init__(self, channels: int, time_size: int) -> None:
        super().__init__()

        self.__to_scale_shift = nn.Sequential(
            nn.Linear(time_size, channels * 2, bias=False),
            nn.LayerNorm(channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def __get_linear_at(self, index: int) -> nn.Linear:
        last_module = self.__to_scale_shift[index]

        if not isinstance(last_module, nn.Linear):
            raise RuntimeError("Can't find last linear module")

        return last_module

    @property
    def last_bias(self) -> th.Tensor:
        last_linear = self.__get_linear_at(-1)

        if last_linear.bias is None:
            raise RuntimeError("Can't find last linear module bias")

        return last_linear.bias

    @property
    def last_weights(self) -> th.Tensor:
        return self.__get_linear_at(-1).weight

    @property
    def first_weights(self) -> th.Tensor:
        return self.__get_linear_at(0).weight

    def forward(self, time_emb: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        proj_time_emb = self.__to_scale_shift(time_emb)
        proj_time_emb = proj_time_emb[:, :, :, None, None]

        scale, shift = th.chunk(proj_time_emb, chunks=2, dim=2)

        return scale, shift
