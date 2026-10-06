from os.path import dirname, join

import pytest


@pytest.fixture(name="wav_path", scope="session")
def get_wav_path() -> str:
    return join(str(dirname(__file__)), "resources", "example_16000Hz.wav")
