import torch as th
import torch.nn as nn


class TimeEmbeder(nn.Embedding):
    def __init__(self, steps: int, size: int) -> None:
        super().__init__(steps, size)


class TimeWrapper(nn.Module):
    def __init__(
        self,
        channels: int,
        time_size: int,
        block: nn.Sequential,
    ) -> None:
        super().__init__()

        self.__to_channels = nn.Sequential(
            nn.Linear(time_size, time_size),
            nn.GELU(),
            TimeBypass(nn.InstanceNorm1d(time_size)),
            nn.Linear(time_size, channels),
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
