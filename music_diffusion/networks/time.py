# -*- coding: utf-8 -*-
import math
from typing import Iterable

import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .convolutions import _BaseConv


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


class TimeWrapper(nn.Module):
    def __init__(
        self,
        time_size: int,
        conv: _BaseConv,
    ) -> None:
        super().__init__()

        self.__block = conv

        channels = conv.out_channels

        self.__to_channels = nn.Sequential(
            weight_norm(nn.Linear(time_size, channels * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(channels * 2, channels * 2)),
        )

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        b, t = x.size()[:2]

        proj_time_emb = self.__to_channels(time_emb)
        proj_time_emb = proj_time_emb[:, :, :, None, None]
        scale, shift = th.chunk(proj_time_emb, chunks=2, dim=2)

        out: th.Tensor = self.__block(x.flatten(0, 1))
        out = th.unflatten(out, 0, (b, t))

        out = out * (scale + 1.0) + shift

        return out


class SequentialTimeWrapper(nn.ModuleList):
    def __init__(
        self,
        time_size: int,
        conv_layers: Iterable[_BaseConv],
    ):
        super().__init__(TimeWrapper(time_size, c) for c in conv_layers)

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        out = x
        for m in self:
            out = m(out, time_emb)
        return out
