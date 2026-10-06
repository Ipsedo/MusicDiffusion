import torch as th


def check_size(
    tensor: th.Tensor,
    batch_size: int,
    step_batch_size: int,
    channels: int,
    img_sizes: tuple[int, int],
) -> None:
    assert tensor.size() == (
        batch_size,
        step_batch_size,
        channels,
        img_sizes[0],
        img_sizes[1],
    )
