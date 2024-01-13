# -*- coding: utf-8 -*-
from os.path import dirname, join

import pytest


@pytest.fixture(name="use_cuda", scope="session")
def get_use_cuda() -> bool:
    return False


@pytest.fixture(name="resources_path", scope="session")
def get_resources_path() -> str:
    return join(dirname(__file__), "resources")


@pytest.fixture(name="wav_path", scope="session")
def get_wav_path(resources_path: str) -> str:
    return join(resources_path, "example_16000Hz.wav")
