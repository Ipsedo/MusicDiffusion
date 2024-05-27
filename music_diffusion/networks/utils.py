# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from torch import nn


class ChannelsModule(ABC, nn.Module):
    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass


class _BaseConv(nn.Sequential, ChannelsModule):
    def __init__(self, out_channels: int, *modules: nn.Module):
        super().__init__(*modules)

        self.__out_channels = out_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels
