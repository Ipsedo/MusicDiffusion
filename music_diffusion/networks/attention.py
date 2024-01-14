# -*- coding: utf-8 -*-
from math import log

import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .convolutions import ChannelProjBlock


def _image_to_seq(x: th.Tensor) -> th.Tensor:
    return x.flatten(2, 3).permute(0, 2, 1)


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__()

        self.__query_conv = ChannelProjBlock(channels, channels)
        self.__key_conv = ChannelProjBlock(channels, key_dim)
        self.__value_conv = ChannelProjBlock(channels, value_dim)

        self.__attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            vdim=value_dim,
            kdim=key_dim,
            batch_first=True,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        _, _, w, h = x.size()

        proj_query = _image_to_seq(self.__query_conv(x))
        proj_key = _image_to_seq(self.__key_conv(x))
        proj_value = _image_to_seq(self.__value_conv(x))

        out: th.Tensor = self.__attention(proj_query, proj_key, proj_value)[0]
        out = out.permute(0, 2, 1)
        out = th.unflatten(out, 2, (w, h))

        out = out + x

        return out


class _PositionalEncoding(nn.Module):
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


class _AutoregTransformer(nn.Module):
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

        self.__pe = _PositionalEncoding(model_dim, target_length)

    def forward(self, y: th.Tensor) -> th.Tensor:
        assert len(y.size()) == 2

        tgt: th.Tensor = self.__start_vec.repeat(y.size(0), 1, 1)
        y = self.__pe(y.unsqueeze(1))

        for _ in range(self.__target_length):
            tgt_next = self.__trf(y, self.__pe(tgt))
            tgt = th.cat([tgt, tgt_next[:, -1, None, :]], dim=1)

        tgt = tgt[:, 1:, :]

        return tgt


class CrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        trf_hidden_dim: int,
        trf_num_heads: int,
        trf_layers: int,
        kv_dim: int,
        kv_length: int,
    ) -> None:
        super().__init__()

        self.__query_conv = ChannelProjBlock(channels, channels)

        self.__cross_att = nn.MultiheadAttention(
            channels,
            1,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True,
        )

        self.__to_channels = nn.Sequential(
            weight_norm(nn.Linear(condition_dim, kv_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(kv_dim * 2, kv_dim * 2)),
        )
        # pylint: disable=unused-private-member
        self.__tau = _AutoregTransformer(
            kv_dim, trf_hidden_dim, trf_num_heads, trf_layers, kv_length
        )
        # pylint: enable=unused-private-member

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        _, _, w, h = x.size()

        proj_query = _image_to_seq(self.__query_conv(x))

        proj_key, proj_value = (
            self.__to_channels(y).unsqueeze(1).chunk(dim=-1, chunks=2)
        )
        # proj_kv = self.__tau(y)

        out: th.Tensor = self.__cross_att(proj_query, proj_key, proj_value)[0]
        out = out.permute(0, 2, 1)
        out = th.unflatten(out, 2, (w, h))

        out = out + x

        return out
