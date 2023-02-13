import torch as th
import torch.nn as nn


# https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, emb_dim: int) -> None:
        super(SelfAttention2d, self).__init__()

        self.__query_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=emb_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        self.__key_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=emb_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        self.__value_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )

        self.__gamma = nn.Parameter(th.zeros(1))

        self.__softmax = nn.Softmax(dim=-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, w, h = x.size()

        # b, c_e, w * h
        proj_query = self.__query_conv(x).view(b, -1, w * h)
        # b, w * h, c_e
        proj_query = proj_query.permute(0, 2, 1)

        # b, c_e, w * h
        proj_key = self.__key_conv(x).view(b, -1, w * h)

        # query(w * h, c_e) @ key(c_e, w * h)
        energy = th.bmm(proj_query, proj_key)

        # b, w * h, w * h
        attention = self.__softmax(energy).permute(0, 2, 1)

        # b, c, w * h
        proj_value = self.__value_conv(x).view(b, -1, w * h)

        out = th.bmm(proj_value, attention)
        out = out.view(b, c, w, h)

        out = self.__gamma * out + x

        return out
