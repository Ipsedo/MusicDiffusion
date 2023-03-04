import torch as th
from torch import nn

from .convolutions import ChannelProjBlock


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        emb_dim: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__()

        self.__query_conv = ChannelProjBlock(channels, emb_dim)
        self.__key_conv = ChannelProjBlock(channels, key_dim)
        self.__value_conv = ChannelProjBlock(channels, value_dim)

        self.__attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            vdim=value_dim,
            kdim=key_dim,
            batch_first=True,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        _, _, w, h = x.size()

        proj_query = SelfAttention2d.__image_to_seq(self.__query_conv(x))
        proj_key = SelfAttention2d.__image_to_seq(self.__key_conv(x))
        proj_value = SelfAttention2d.__image_to_seq(self.__value_conv(x))

        out: th.Tensor = self.__attention(proj_query, proj_key, proj_value)[0]
        out = out.permute(0, 2, 1)
        out = th.unflatten(out, 2, (w, h))

        out = out + x

        return out

    @staticmethod
    def __image_to_seq(x: th.Tensor) -> th.Tensor:
        return x.flatten(2, 3).permute(0, 2, 1)
