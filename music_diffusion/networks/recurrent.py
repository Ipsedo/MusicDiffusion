# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .tau import PositionalEncoding


class ToTimeSeries(nn.Module):
    def __init__(self, channels: int, output_dim: int) -> None:
        super().__init__()

        self.__to_output = nn.Sequential(
            weight_norm(nn.Linear(channels, output_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(output_dim * 2, output_dim)),
            nn.Mish(),
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


class MiddleTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        trf_dim: int,
        hidden_dim: int,
        tau_dim: int,
        num_heads: int,
        max_length: int,
    ) -> None:
        super().__init__()

        self.__to_time_series = ToTimeSeries(channels, trf_dim)

        self.__to_start_token = nn.Sequential(
            weight_norm(nn.Linear(tau_dim, trf_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(trf_dim * 2, trf_dim)),
            nn.Mish(),
        )

        self.__trf = nn.Transformer(
            trf_dim,
            nhead=num_heads,
            num_decoder_layers=3,
            num_encoder_layers=3,
            dim_feedforward=hidden_dim,
            activation=nn.functional.gelu,
            batch_first=True,
        )

        self.__to_channels = weight_norm(nn.Linear(trf_dim, channels * 2))

        self.__pe = PositionalEncoding(trf_dim, max_length)

    def forward(self, x: th.Tensor, y_encoded: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__to_time_series(x)
        out = self.__pe(out)

        tgt = self.__to_start_token(y_encoded).unsqueeze(1)

        for _ in range(out.size(1)):
            tgt_next = self.__trf(out, self.__pe(tgt))
            tgt = th.cat([tgt, tgt_next[:, -1, None, :]], dim=1)

        out = tgt[:, 1:, :]

        shift, scale = (
            self.__to_channels(out)[:, None, :, :]
            .permute(0, 3, 1, 2)
            .chunk(dim=1, chunks=2)
        )

        out = x * (scale + 1) + shift

        return out
