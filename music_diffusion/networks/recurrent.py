# -*- coding: utf-8 -*-

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .utils import Agg, Permute


class AggregateFrequencies(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()

        self.__to_input = nn.Linear(input_dim, hidden_dim)
        self.__key = nn.Parameter(th.rand(1, hidden_dim, 1))
        self.__to_value = nn.Linear(input_dim, output_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, w, h = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * h, w, c)

        q = self.__to_input(x)
        k = self.__key.repeat(b * h, 1, 1)
        v = self.__to_value(x)

        weight = F.softmax(th.bmm(q, k), dim=1).transpose(1, 2)

        out = th.bmm(weight, v).view(b, h, -1)

        return out


class MiddleRecurrent(nn.Module):
    def __init__(
        self, channels: int, lstm_dim: int, hidden_dim: int, tau_dim: int
    ) -> None:
        super().__init__()

        self.__to_time_series = nn.Sequential(
            Permute(0, 2, 3, 1),
            weight_norm(nn.Linear(channels, lstm_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(lstm_dim * 2, lstm_dim)),
            nn.Mish(),
            Agg("max", 1),
        )

        self.__to_h_and_c = nn.Sequential(
            weight_norm(nn.Linear(tau_dim, hidden_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(hidden_dim * 2, hidden_dim * 2)),
            nn.Mish(),
        )

        self.__lstm = nn.LSTM(lstm_dim, hidden_dim, batch_first=True)

        self.__to_channels = weight_norm(nn.Linear(hidden_dim, channels * 2))

    def forward(self, x: th.Tensor, y_encoded: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__to_time_series(x)
        h, c = (
            self.__to_h_and_c(y_encoded).unsqueeze(0).chunk(dim=-1, chunks=2)
        )

        out, _ = self.__lstm(out, (h.contiguous(), c.contiguous()))
        shift, scale = (
            self.__to_channels(out)[:, None, :, :]
            .permute(0, 3, 1, 2)
            .chunk(dim=1, chunks=2)
        )

        out = x * (scale + 1) + shift

        return out
