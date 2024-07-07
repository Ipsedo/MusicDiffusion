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

        self.__w_b = nn.Parameter(th.ones(in_features, out_features, 1))
        self.__w_s = nn.Parameter(th.ones(in_features, out_features, 1))
        self.__c = nn.Parameter(
            th.ones(in_features, out_features, self.__act_fun.get_size())
        )

        xavier_normal_(self.__w_b)
        normal_(self.__c, 0, 1e-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # output dim
        x = x.unsqueeze(-1)

        return th.sum(
            self.__w_b * self.__res_act_fun(x).unsqueeze(-1)
            + self.__w_s * self.__c * self.__act_fun(x),
            dim=[-3, -1],  # sum over input and activation spaces
        )
