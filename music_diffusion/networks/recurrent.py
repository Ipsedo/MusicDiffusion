# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class ToTimeSeries(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()

        self.__to_output = nn.Sequential(
            weight_norm(nn.Linear(channels, output_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(output_dim * 2, output_dim)),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = x.permute(0, 2, 3, 1)
        out = self.__to_output(out)
        out = out.sum(dim=1)

        return out


class MiddleRecurrent(nn.Module):
    def __init__(
        self, channels: int, lstm_dim: int, hidden_dim: int, tau_dim: int
    ) -> None:
        super().__init__()

        self.__to_time_series = ToTimeSeries(channels, lstm_dim)

        self.__to_h_and_c = nn.Sequential(
            weight_norm(nn.Linear(tau_dim, hidden_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(hidden_dim * 2, hidden_dim * 2)),
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
