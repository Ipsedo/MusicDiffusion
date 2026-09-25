from os.path import dirname, join

import pytest
import torch as th


@pytest.fixture(name="device", scope="session")
def get_device() -> th.device:
    return th.device("cpu")


@pytest.fixture(name="wav_path", scope="session")
def get_wav_path() -> str:
    return join(str(dirname(__file__)), "resources", "example_16000Hz.wav")
