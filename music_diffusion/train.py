# -*- coding: utf-8 -*-
from statistics import mean
from typing import NamedTuple, Optional

import mlflow
import torch as th
from torch.nn import functional as th_f
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .data import AudioDataset, ChangeType, ChannelMinMaxNorm, RangeChange
from .networks import Denoiser, Noiser
from .utils import ModelOptions, Saver

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
        ("vlb_loss_factor", float),
        ("noiser_state_dict", Optional[str]),
        ("denoiser_state_dict", Optional[str]),
        ("optim_state_dict", Optional[str]),
    ],
)


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:

    mlflow.set_experiment("music_diffusion")

    with mlflow.start_run(run_name=train_options.run_name):

        if model_options.cuda:
            th.backends.cudnn.benchmark = True

        noiser = Noiser(
            model_options.steps,
            model_options.beta_1,
            model_options.beta_t,
        )

        # pylint: disable=duplicate-code
        denoiser = Denoiser(
            model_options.input_channels,
            model_options.steps,
            model_options.time_size,
            model_options.beta_1,
            model_options.beta_t,
            model_options.unet_channels,
            model_options.use_attentions,
            model_options.attention_heads,
            model_options.norm_groups,
        )
        # pylint: enable=duplicate-code

        print(f"Parameters count = {denoiser.count_parameters()}")

        if model_options.cuda:
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
            model_options.input_channels,
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
                "beta_1": model_options.beta_1,
                "beta_t": model_options.beta_t,
                "steps": model_options.steps,
                "time_size": model_options.time_size,
                "input_channels": model_options.input_channels,
                "unet_channels": model_options.unet_channels,
                "input_dataset": train_options.dataset_path,
            }
        )

        device = "cuda" if model_options.cuda else "cpu"

        losses = [0.0 for _ in range(train_options.metric_window)]
        metric_step = 0

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for x_0 in tqdm_bar:

                if model_options.cuda:
                    x_0 = x_0.cuda()

                x_0 = transform(x_0)

                t = th.randint(
                    0,
                    model_options.steps,
                    (
                        train_options.batch_size,
                        train_options.step_batch_size,
                    ),
                    device=device,
                )

                x_t, eps = noiser(x_0, t)
                eps_theta, v_theta = denoiser(x_t, t)

                t_prev = t - 1
                t_prev[t_prev < 0] = 0
                x_t_prev, _ = noiser(x_0, t_prev, eps)

                loss_simple = th.pow(eps - eps_theta, 2.0).mean()

                prior = denoiser.prior(
                    x_t_prev,
                    x_t,
                    t,
                    eps_theta.detach(),
                    v_theta,
                )
                posterior = noiser.posterior(x_t_prev, x_t, x_0, t)

                loss_vlb = th_f.kl_div(
                    prior.flatten(0, 1),
                    posterior.flatten(0, 1),
                    reduction="batchmean",
                )

                loss = loss_simple + train_options.vlb_loss_factor * loss_vlb

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                del losses[0]
                losses.append(loss.item())

                mlflow.log_metric("loss", loss.item(), step=metric_step)
                metric_step += 1

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1} - "
                    f"save {saver.curr_save} "
                    f"[{saver.curr_step} / {train_options.save_every - 1}] "
                    f"loss = {mean(losses):.6f}"
                )

                saver.save()
