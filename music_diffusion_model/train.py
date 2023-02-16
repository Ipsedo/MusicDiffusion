from statistics import mean
from typing import List, NamedTuple, Optional, Tuple

import mlflow
import torch as th
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .data import AudioDataset, ChangeType, ChannelMinMaxNorm, RangeChange
from .networks import Denoiser, Noiser
from .utils import Saver

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("run_name", str),
        ("dataset_path", str),
        ("batch_size", int),
        ("step_batch_size", int),
        ("epochs", int),
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("unet_channels", List[Tuple[int, int]]),
        ("use_attentions", List[bool]),
        ("attention_heads", int),
        ("time_size", int),
        ("cuda", bool),
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


def train(train_options: TrainOptions) -> None:

    mlflow.set_experiment("music_diffusion_model")

    with mlflow.start_run(run_name=train_options.run_name):

        if train_options.cuda:
            th.backends.cudnn.benchmark = True

        noiser = Noiser(
            train_options.steps,
            train_options.beta_1,
            train_options.beta_t,
        )

        denoiser = Denoiser(
            train_options.input_channels,
            train_options.steps,
            train_options.time_size,
            train_options.beta_1,
            train_options.beta_t,
            train_options.unet_channels,
            train_options.use_attentions,
            train_options.attention_heads,
        )

        if train_options.cuda:
            noiser.cuda()
            denoiser.cuda()

        optim = th.optim.Adam(
            denoiser.parameters(),
            lr=train_options.learning_rate,
        )

        if train_options.noiser_state_dict is not None:
            noiser.load_state_dict(th.load(train_options.noiser_state_dict))
        if train_options.denoiser_state_dict is not None:
            denoiser.load_state_dict(
                th.load(train_options.denoiser_state_dict)
            )
        if train_options.optim_state_dict is not None:
            optim.load_state_dict(th.load(train_options.optim_state_dict))

        saver = Saver(
            train_options.input_channels,
            noiser,
            denoiser,
            optim,
            train_options.output_directory,
            train_options.save_every,
            train_options.nb_samples,
        )

        dataset = AudioDataset(train_options.dataset_path)

        dataloader = DataLoader(
            dataset,
            batch_size=train_options.batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
            pin_memory=True,
        )

        transform = Compose(
            [
                ChangeType(th.float),
                ChannelMinMaxNorm(),
                RangeChange(-1.0, 1.0),
            ]
        )

        mlflow.log_params(
            {
                "batch_size": train_options.batch_size,
                "step_batch_size": train_options.step_batch_size,
                "learning_rate": train_options.learning_rate,
                "epochs": train_options.epochs,
                "beta_1": train_options.beta_1,
                "beta_t": train_options.beta_t,
                "steps": train_options.steps,
                "time_size": train_options.time_size,
                "input_channels": train_options.input_channels,
                "unet_channels": train_options.unet_channels,
                "input_dataset": train_options.dataset_path,
            }
        )

        device = "cuda" if train_options.cuda else "cpu"

        losses = [0.0 for _ in range(train_options.metric_window)]
        metric_step = 0

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for x in tqdm_bar:

                if train_options.cuda:
                    x = x.cuda()

                x = transform(x)

                t = th.randint(
                    0,
                    train_options.steps,
                    (
                        train_options.batch_size,
                        train_options.step_batch_size,
                    ),
                    device=device,
                )

                x_noised, eps = noiser(x, t)
                eps_theta = denoiser(x_noised, t)

                loss = th.pow(eps - eps_theta, 2.0)
                loss = loss.mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                del losses[0]
                losses.append(loss.item())

                mlflow.log_metric("loss", loss.item(), step=metric_step)
                metric_step += 1

                saver.save()

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1} - "
                    f"save {saver.curr_save} "
                    f"[{saver.curr_step} / {train_options.save_every - 1}] "
                    f"loss = {mean(losses):.4f}"
                )
