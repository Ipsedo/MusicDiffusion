# -*- coding: utf-8 -*-
from typing import List, NamedTuple, Optional, Tuple

ModelOptions = NamedTuple(
    "ModelOptions",
    [
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("unet_channels", List[Tuple[int, int]]),
        ("time_size", int),
        ("norm_groups", int),
        ("cuda", bool),
    ],
)

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("run_name", str),
        ("dataset_path", str),
        ("batch_size", int),
        ("step_batch_size", int),
        ("epochs", int),
        ("learning_rate", float),
        ("metric_window", int),
        ("save_every", int),
        ("output_directory", str),
        ("nb_samples", int),
        ("noiser_state_dict", Optional[str]),
        ("denoiser_state_dict", Optional[str]),
        ("optim_state_dict", Optional[str]),
    ],
)

GenerateOptions = NamedTuple(
    "GenerateOptions",
    [
        ("fast_sample", Optional[int]),
        ("denoiser_dict_state", str),
        ("ema_denoiser", bool),
        ("output_dir", str),
        ("frames", int),
        ("musics", int),
    ],
)
