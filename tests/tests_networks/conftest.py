import pytest
import torch as th


@pytest.fixture(name="device", scope="session")
def get_device() -> th.device:
    return th.device("cpu")
