# -*- coding: utf-8 -*-
from typing import Callable

import torch as th
from torch import nn
from torch.nn.init import normal_, xavier_normal_

from .activations import ActivationFunction


class LinearKAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self._w_b = nn.Parameter(th.ones(out_features, in_features))
        self._w_s = nn.Parameter(th.ones(out_features, in_features))
        self._c = nn.Parameter(
            th.ones(self.__act_fun.get_size(), out_features, in_features)
        )

        xavier_normal_(self._w_b)
        normal_(self._c, 0, 1e-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2

        return th.sum(
            self._w_s * th.einsum("bai,aoi->boi", self.__act_fun(x), self._c)
            + self._w_b * self.__res_act_fun(x).unsqueeze(1),
            dim=2,
        )
