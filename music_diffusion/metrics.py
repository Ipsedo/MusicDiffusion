# -*- coding: utf-8 -*-
from statistics import mean
from typing import List, Union

import torch as th


class Metric:
    def __init__(self, window_size: int) -> None:
        self.__window_size = window_size
        self.__result: List[float] = [0.0]

    def add_result(self, res: Union[th.Tensor, float]) -> None:
        if isinstance(res, th.Tensor):
            if len(res.size()) >= 1:
                for t in res:
                    self.__result.append(th.mean(t).item())
            else:
                self.__result.append(res.item())
        else:
            self.__result.append(res)

        while len(self.__result) > self.__window_size:
            self.__result.pop(0)

    def get_smoothed_metric(self) -> float:
        return mean(self.__result)

    def get_last_metric(self) -> float:
        return self.__result[-1]
