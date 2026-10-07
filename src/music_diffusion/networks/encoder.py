import torch as th
from torch import nn

from .convolutions import StrideConvBlock


class ChunkEncoder(nn.Module):
    """Map a clean chunk (B, C, H, W) to a global conditioning vector z.

    The vector is trained in a cross-chunk fashion : the encoder sees one
    chunk of a piece while the denoiser reconstructs another one, so z can
    only hold what is invariant across the piece (instruments, register,
    tempo, texture) and not the content of any specific chunk.
    """

    def __init__(
        self,
        channels: list[tuple[int, int]],
        group_norm_nums: list[int],
        z_size: int,
    ) -> None:
        super().__init__()

        assert len(channels) == len(group_norm_nums)
        assert all(
            channels[i][1] == channels[i + 1][0]
            for i in range(len(channels) - 1)
        )

        self.__conv = nn.Sequential(
            *(
                StrideConvBlock(c_i, c_o, g, "down")
                for (c_i, c_o), g in zip(channels, group_norm_nums)
            )
        )

        self.__to_z = nn.Sequential(
            nn.Linear(channels[-1][1], z_size),
            nn.LayerNorm(z_size),
        )

        self.__z_size = z_size

    @property
    def z_size(self) -> int:
        return self.__z_size

    def forward(self, x_0: th.Tensor) -> th.Tensor:
        assert len(x_0.size()) == 4

        out: th.Tensor = self.__conv(x_0)
        out = out.mean(dim=(2, 3))

        z: th.Tensor = self.__to_z(out)

        return z
