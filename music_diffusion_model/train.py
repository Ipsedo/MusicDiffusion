from os import mkdir
from os.path import exists, isdir
from statistics import mean
from typing import NamedTuple

import matplotlib.pyplot as plt
import mlflow
import torch as th
import torch.nn.functional as th_f
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Pad
from tqdm import tqdm

from .data import ChangeType, ChannelMinMaxNorm, MNISTDataset, RangeChange
from .networks import Denoiser, Noiser

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("run_name", str),
        ("batch_size", int),
        ("step_batch_size", int),
        ("epochs", int),
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("time_size", int),
        ("cuda", bool),
        ("learning_rate", float),
        ("metric_window", int),
        ("save_every", int),
        ("output_directory", str),
    ],
)


def train(train_options: TrainOptions) -> None:

    if not exists(train_options.output_directory):
        mkdir(train_options.output_directory)
    elif not isdir(train_options.output_directory):
        raise NotADirectoryError(train_options.output_directory)

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
        )

        if train_options.cuda:
            noiser.cuda()
            denoiser.cuda()

        optim = th.optim.Adam(
            denoiser.parameters(),
            lr=train_options.learning_rate,
        )

        dataset = MNISTDataset()

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
                Pad(2, 0, "constant"),
                ChangeType(th.float),
                ChannelMinMaxNorm(),
                RangeChange(-1.0, 1.0),
            ]
        )

        mlflow.log_params(
            {
                "batch_size": train_options.batch_size,
                "learning_rate": train_options.learning_rate,
                "epochs": train_options.epochs,
                "beta_1": train_options.beta_1,
                "beta_t": train_options.beta_t,
                "steps": train_options.steps,
                "time_size": train_options.time_size,
                "input_channels": train_options.input_channels,
            }
        )

        device = "cuda" if train_options.cuda else "cpu"

        losses = [0.0 for _ in range(train_options.metric_window)]

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
                x_noised = th.clip(x_noised, -1.0, 1.0)
                eps_theta = denoiser(x_noised, t)

                loss = th_f.mse_loss(eps_theta, eps, reduction="none")
                loss = loss * denoiser.loss_scale(t)
                loss = loss.sum(dim=[2, 3, 4]).mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                del losses[0]
                losses.append(loss.item())

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1}, loss = {mean(losses):.4f}"
                )

                mlflow.log_metric("loss", loss.item())

            z = th.randn(
                1, train_options.input_channels, 32, 32, device=device
            )
            o = denoiser.sample(z)

            plt.matshow(o[0, 0].detach().cpu(), cmap="Greys")
            plt.title(f"Epoch {e}")
            plt.show()
