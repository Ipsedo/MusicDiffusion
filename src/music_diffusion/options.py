from pydantic import BaseModel

from .networks import Denoiser, Noiser


class ModelOptions(BaseModel):
    steps: int
    unet_channels: list[tuple[int, int]]
    unet_group_norm_nums: list[int]
    time_size: int
    cuda: bool

    def new_denoiser(self) -> Denoiser:
        return Denoiser(
            self.steps,
            self.time_size,
            self.unet_channels,
            self.unet_group_norm_nums,
        )

    def new_noiser(self) -> Noiser:
        return Noiser(self.steps)


class TrainOptions(BaseModel):
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
    noiser_state_dict: str | None
    denoiser_state_dict: str | None
    ema_state_dict: str | None
    optim_state_dict: str | None


class GenerateOptions(BaseModel):
    fast_sample: int | None
    denoiser_dict_state: str
    ema_denoiser: bool
    output_dir: str
    frames: int
    musics: int
