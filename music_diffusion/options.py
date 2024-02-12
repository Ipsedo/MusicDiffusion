# -*- coding: utf-8 -*-
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch as th

from .data.metadata import multi_label_one_hot_encode, one_hot_encode
from .networks import Denoiser, Noiser


class ModelOptions(NamedTuple):
    steps: int
    unet_channels: List[Tuple[int, int]]
    time_size: int
    lstm_dim: int
    lstm_hidden_dim: int
    tau_nb_key: int
    tau_nb_scoring: int
    tau_hidden_dim: int
    tau_layers: int
    cuda: bool

    def new_denoiser(self) -> Denoiser:
        return Denoiser(
            self.steps,
            self.time_size,
            self.unet_channels,
            self.lstm_dim,
            self.lstm_hidden_dim,
            self.tau_nb_key,
            self.tau_nb_scoring,
            self.tau_hidden_dim,
            self.tau_layers,
        )

    def new_noiser(self) -> Noiser:
        return Noiser(self.steps)


class TrainOptions(NamedTuple):
    run_name: str
    dataset_path: str
    batch_size: int
    step_batch_size: int
    epochs: int
    learning_rate: float
    metric_window: int
    save_every: int
    output_directory: str
    nb_samples: int
    noiser_state_dict: Optional[str]
    denoiser_state_dict: Optional[str]
    ema_state_dict: Optional[str]
    optim_state_dict: Optional[str]


class GenerateOptions(NamedTuple):
    fast_sample: Optional[int]
    denoiser_dict_state: str
    ema_denoiser: bool
    output_dir: str
    frames: int
    musics: int
    magn_scale: float
    keys: List[str]
    scoring_list: List[List[str]]
    key_to_idx: Dict[str, int]
    scoring_to_idx: Dict[str, int]

    def get_y_key(self) -> th.Tensor:
        key = th.stack(
            [one_hot_encode(k, self.key_to_idx) for k in self.keys], dim=0
        )

        return key

    def get_y_scoring(self) -> th.Tensor:
        scoring = th.stack(
            [
                multi_label_one_hot_encode(s_l, self.scoring_to_idx)
                for s_l in self.scoring_list
            ],
            dim=0,
        )

        return scoring
