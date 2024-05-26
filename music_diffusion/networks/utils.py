# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from torch import nn


class ChannelModule(ABC, nn.Module):
    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass
