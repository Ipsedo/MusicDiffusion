from typing import NamedTuple

import matplotlib.pyplot as plt
import mlflow
import torch as th
import torch.nn.functional as th_f
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MNISTDataset
from .networks import Denoiser, Noiser

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("run_name", str),
        ("batch_size", int),
        ("epochs", int),
        ("steps", int),
        ("beta_1", float),
        ("beta_t", float),
        ("input_channels", int),
        ("time_size", int),
        ("cuda", bool),
        ("learning_rate", float),
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
        )

        if train_options.cuda:
            noiser.cuda()
            denoiser.cuda()

        optim = th.optim.Adam(
            denoiser.parameters(), lr=train_options.learning_rate
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

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for x in tqdm_bar:
                if train_options.cuda:
                    x = x.cuda()

                x_noised, eps = noiser(x)
                eps_theta = denoiser(x_noised)

                loss = th_f.mse_loss(eps_theta, eps, reduction="none")
                loss = loss * denoiser.loss_scale
                loss = loss.sum(dim=1).mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1}, loss = {loss.item():.4f}"
                )

                mlflow.log_metric("loss", loss.item())

            z = th.randn(1, train_options.input_channels, 32, 32)

            if train_options.cuda:
                z = z.cuda()

            o = denoiser.sample(z)

            plt.matshow(o[0, 0].detach().cpu(), cmap="Greys")
            plt.title(f"Epoch {e}")
            plt.show()
