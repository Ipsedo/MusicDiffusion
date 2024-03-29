# -*- coding: utf-8 -*-
from os.path import dirname, join

import pytest


@pytest.fixture(name="use_cuda", scope="session")
def use_cuda() -> bool:
    return False


@pytest.fixture(name="wav_path", scope="session")
def wav_path() -> str:
    return join(dirname(__file__), "resources", "example_16000Hz.wav")
