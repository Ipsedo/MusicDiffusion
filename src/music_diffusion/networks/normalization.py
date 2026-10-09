import torch as th
from torch import nn
from torch.nn import functional as th_f


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-12) -> None:
        super().__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x / th.sqrt(
            x.pow(2.0).mean(dim=1, keepdim=True) + self.__epsilon
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class LayerNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True)

        return (x - mean) / th.sqrt(var + self.__epsilon)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.__epsilon})"

    def __str__(self) -> str:
        return self.__repr__()


class RowGroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        num_neighbors: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels must by divisible by num_groups "
                f"(channels={num_channels}, groups={num_groups})"
            )

        self.__num_groups = num_groups
        self.__num_channels = num_channels

        self.__num_neighbors = num_neighbors

        self.__eps = eps

        self.affine = affine

        if affine:
            self.weight = nn.Parameter(th.ones(num_channels))
            self.bias = nn.Parameter(th.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()
        g, k = self.__num_groups, self.__num_neighbors

        xg = x.reshape(b, g, c // g, h, w)
        xf = xg.float()

        # per time step statistics : over the group channels and frequencies
        mean_col = xf.mean(dim=(2, 3))
        sq_col = (xf * xf).mean(dim=(2, 3))

        stats = th.stack([mean_col, sq_col], dim=2).reshape(b * g * 2, 1, w)

        # pylint: disable=not-callable
        # smooth the statistics over the neighboring time steps
        stats = th_f.avg_pool1d(
            stats,
            kernel_size=2 * k + 1,
            stride=1,
            padding=k,
            count_include_pad=False,
        )

        mean, sq = stats.reshape(b, g, 2, w).unbind(dim=2)
        var = (sq - mean * mean).clamp_min(0.0)

        mean = mean.reshape(b, g, 1, 1, w)
        var = var.reshape(b, g, 1, 1, w)

        y = (xf - mean) * th.rsqrt(var + self.__eps)
        y = y.reshape(b, c, h, w).to(x.dtype)

        if self.affine:
            y = y * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)

        return y

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.__num_groups}, "
            f"num_channels={self.__num_channels}, "
            f"num_neighbors={self.__num_neighbors}, "
            f"eps={self.__eps}, "
            f"affine={self.affine}"
        )
