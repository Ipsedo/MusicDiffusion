# -*- coding: utf-8 -*-
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
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

        self.__to_key_value = nn.Sequential(
            weight_norm(nn.Linear(condition_dim, kv_dim * kv_length)),
            nn.Mish(),
            weight_norm(nn.Linear(kv_dim * kv_length, kv_dim * kv_length * 2)),
        )

        self.__kv_length = kv_length
        self.__kv_dim = kv_dim

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        _, _, w, h = x.size()

        proj_query = _image_to_seq(self.__query_conv(x))
        proj_key, proj_value = th.chunk(
            self.__to_key_value(y).view(
                -1, self.__kv_length, self.__kv_dim * 2
            ),
            dim=-1,
            chunks=2,
        )

        out: th.Tensor = self.__cross_att(proj_query, proj_key, proj_value)[0]
        out = out.permute(0, 2, 1)
        out = th.unflatten(out, 2, (w, h))

        out = out + x

        return out
