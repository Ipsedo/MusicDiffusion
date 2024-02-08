# -*- coding: utf-8 -*-
from math import log

import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, length: int):
        super().__init__()

        position = th.arange(length).unsqueeze(1)

        div_term = th.exp(
            th.arange(0, model_dim, 2) * (-log(10000.0) / model_dim)
        )

        pe = th.zeros(1, length, model_dim)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)

        self.register_buffer("_pe", pe)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = self._pe[:, : x.size(1), :] + x
        return out


class AutoregTransformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        layers: int,
        target_length: int,
    ) -> None:
        super().__init__()

        self.__trf = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_decoder_layers=layers,
            num_encoder_layers=layers,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
        )

        self.__target_length = target_length

        self.__start_vec = nn.Parameter(th.randn((1, 1, model_dim)))

        self.__pe = PositionalEncoding(model_dim, target_length)

    def forward(self, y: th.Tensor) -> th.Tensor:
        assert len(y.size()) == 2

        tgt: th.Tensor = self.__start_vec.repeat(y.size(0), 1, 1)
        y = self.__pe(y.unsqueeze(1))

        for _ in range(self.__target_length):
            tgt_next = self.__trf(y, self.__pe(tgt))
            tgt = th.cat([tgt, tgt_next[:, -1, None, :]], dim=1)

        tgt = tgt[:, 1:, :]

        return tgt


class ConditionEncoder(nn.Sequential):
    def __init__(
        self, condition_dim: int, hidden_dim: int, nb_layers: int
    ) -> None:
        assert nb_layers > 0
        layers_dim = [
            (condition_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(nb_layers)
        ]

        super().__init__(
            *[
                nn.Sequential(
                    weight_norm(nn.Linear(d_i, d_o)),
                    nn.Mish(),
                )
                for d_i, d_o in layers_dim
            ]
        )
