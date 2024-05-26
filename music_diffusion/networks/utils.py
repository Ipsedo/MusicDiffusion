# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from torch import nn


class ChannelsModule(ABC, nn.Module):
    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass
