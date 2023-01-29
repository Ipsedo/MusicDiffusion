import torch as th
import torch.nn as nn

from .unet import UNet


class TimeEmbeder(nn.Embedding):
    def __init__(self, steps: int, size: int) -> None:
        super().__init__(steps, size)


class TimeWrapper(nn.Module):
    def __init__(
        self,
        unet: UNet,
        steps: int,
        time_size: int,
    ) -> None:
        super().__init__()

        self.__emb = TimeEmbeder(steps, time_size)

        self.__unet = unet

    def forward(self, x: th.Tensor, time: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 5
        assert len(time.size()) == 2
        assert x.size(0) == time.size(0)

        b, t, _, w, h = x.size()

        time_vec = self.__emb(time)
        time_vec = time_vec[:, :, :, None, None].repeat(1, 1, 1, w, h)
        x_time = th.cat([x, time_vec], dim=2)

        x_time = x_time.flatten(0, 1)
        res: th.Tensor = self.__unet(x_time)
        res = res.view(b, t, -1, w, h)

        return res
