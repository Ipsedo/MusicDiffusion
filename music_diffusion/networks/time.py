import math

import torch as th
import torch.nn as nn


class TimeEmbeder(nn.Module):
    def __init__(self, steps: int, size: int) -> None:
        super().__init__()

        pos_emb = th.zeros(steps, size)
        position = th.arange(0, steps).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, size, 2, dtype=th.float) * (-math.log(10000.0) / size)
        )
        pos_emb[:, 0::2] = th.sin(position.float() * div_term)
        pos_emb[:, 1::2] = th.cos(position.float() * div_term)

        self.pos_emb: th.Tensor

        self.register_buffer("pos_emb", pos_emb)

    def forward(self, t_index: th.Tensor) -> th.Tensor:
        b, t = t_index.size()

        t_index = t_index.flatten()

        out = th.index_select(self.pos_emb, dim=0, index=t_index)
        out = th.unflatten(out, 0, (b, t))

        return out


class TimeWrapper(nn.Module):
    def __init__(
        self,
        channels: int,
        time_size: int,
        block: nn.Sequential,
    ) -> None:
        super().__init__()

        self.__to_channels = nn.Sequential(
            nn.Linear(time_size, channels),
            nn.ELU(),
            TimeBypass(nn.BatchNorm1d(channels)),
        )

        self.__block = block

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        b, t, _, _, _ = x.size()

        time_emb = self.__to_channels(time_emb)
        time_emb = time_emb[:, :, :, None, None]
        x_time = x + time_emb

        x_time = x_time.flatten(0, 1)
        out: th.Tensor = self.__block(x_time)
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
