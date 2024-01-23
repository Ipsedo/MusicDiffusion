# -*- coding: utf-8 -*-

import torch as th
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .convolutions import ChannelProjBlock
from .utils import Permute, View


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
        tau_hidden_dim: int,
        kv_dim: int,
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
            weight_norm(nn.Linear(tau_hidden_dim, kv_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(kv_dim * 2, kv_dim * 2)),
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        _, _, w, h = x.size()

        proj_query = _image_to_seq(self.__query_conv(x))

        proj_key, proj_value = (
            self.__to_channels(y).unsqueeze(1).chunk(dim=-1, chunks=2)
        )

        out: th.Tensor = self.__cross_att(proj_query, proj_key, proj_value)[0]
        out = out.permute(0, 2, 1)
        out = th.unflatten(out, 2, (w, h))

        return out


class BoxCrossAttention(nn.Module):
    def __init__(
        self, channels: int, tau_hidden_dim: int, kv_dim: int
    ) -> None:
        super().__init__()

        self.__kernel_size = 3

        self.__query_conv = nn.Sequential(
            ChannelProjBlock(channels, channels),
            nn.Unfold(kernel_size=self.__kernel_size, stride=1, padding=1),
            View(channels, self.__kernel_size**2, -1),
            Permute(0, 3, 2, 1),
            nn.Flatten(0, 1),
        )

        self.__cross_att = nn.MultiheadAttention(
            channels,
            1,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True,
        )

        self.__to_key_value = nn.Sequential(
            weight_norm(nn.Linear(tau_hidden_dim, kv_dim * 2)),
            nn.Mish(),
            weight_norm(nn.Linear(kv_dim * 2, kv_dim * 2)),
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        b, c, w, h = x.size()

        proj_query = self.__query_conv(x)

        proj_key, proj_value = (
            self.__to_key_value(y)
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, w * h, 1, 1)
            .flatten(0, 1)
            .chunk(dim=-1, chunks=2)
        )

        out: th.Tensor = (
            self.__cross_att(proj_query, proj_key, proj_value)[0]
            .view(b, w * h, self.__kernel_size**2, c)
            .permute(0, 3, 2, 1)
            .sum(dim=2)
            .view(b, c, w, h)
        )

        return out
