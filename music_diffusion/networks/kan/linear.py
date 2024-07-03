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

        self.__in_features = in_features

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self._w = nn.Parameter(th.ones(in_features, out_features))
        self._c = nn.Parameter(
            th.ones(in_features, out_features, self.__act_fun.get_size())
        )

        xavier_normal_(self._w, 1e-1)
        normal_(self._c, 0, 1e-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.size(-1) == self.__in_features

        # output dim
        x = x.unsqueeze(-1)

        return th.sum(
            self._w
            * (
                self.__res_act_fun(x)
                + th.sum(self._c * self.__act_fun(x), dim=-1)
            ),
            -2,  # sum over input space
        )
