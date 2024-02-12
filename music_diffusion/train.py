# -*- coding: utf-8 -*-

import mlflow
import torch as th
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ConditionAudioDataset
from .metrics import Metric
from .networks import mse, normal_kl_div
from .options import ModelOptions, TrainOptions
from .saver import Saver


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:

    mlflow.set_experiment("music_diffusion")

    with mlflow.start_run(run_name=train_options.run_name):

        if model_options.cuda:
            th.backends.cudnn.benchmark = True

        noiser = model_options.new_noiser()
        denoiser = model_options.new_denoiser()
        denoiser_ema = EMA(denoiser, include_online_model=True)

        print(f"Parameters count = {denoiser.count_parameters()}")

        if model_options.cuda:
            noiser.cuda()
            denoiser.cuda()
            denoiser_ema.cuda()

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
        if train_options.ema_state_dict is not None:
            denoiser_ema.load_state_dict(th.load(train_options.ema_state_dict))
        if train_options.optim_state_dict is not None:
            optim.load_state_dict(th.load(train_options.optim_state_dict))

        saver = Saver(
            model_options.unet_channels[0][0],
            noiser,
            denoiser,
            optim,
            denoiser_ema,
            train_options.output_directory,
            train_options.save_every,
            train_options.nb_samples,
            train_options.dataset_path,
        )

        dataset = ConditionAudioDataset(train_options.dataset_path)

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
                "step_batch_size": train_options.step_batch_size,
                "learning_rate": train_options.learning_rate,
                "epochs": train_options.epochs,
                "steps": model_options.steps,
                "time_size": model_options.time_size,
                "unet_channels": model_options.unet_channels,
                "tau_nb_key": model_options.tau_nb_key,
                "tau_nb_scoring": model_options.tau_nb_scoring,
                "tau_hidden_dim": model_options.tau_hidden_dim,
                "tau_layers": model_options.tau_layers,
                "input_dataset": train_options.dataset_path,
            }
        )

        device = "cuda" if model_options.cuda else "cpu"

        losses = Metric(train_options.metric_window)
        mse_losses = Metric(train_options.metric_window)
        # vlb_losses = Metric(train_options.metric_window)
        kl_losses = Metric(train_options.metric_window)
        # nll_losses = Metric(train_options.metric_window)
        grad_norms = Metric(train_options.metric_window)

        metric_step = 0

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for x_0, y_key, y_scoring in tqdm_bar:

                if model_options.cuda:
                    x_0 = x_0.cuda()
                    y_key = y_key.cuda()
                    y_scoring = y_scoring.cuda()

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
                eps_theta, v_theta = denoiser(x_t, t, y_key, y_scoring)

                loss_mse = mse(eps, eps_theta)

                q_mu, q_var = noiser.posterior(x_t, x_0, t)
                p_mu, p_var = denoiser.prior(
                    x_t, t, eps_theta.detach(), v_theta
                )

                loss_kl = normal_kl_div(q_mu, q_var, p_mu, p_var)
                # loss_nll = negative_log_likelihood(x_0, p_mu, p_var)
                # loss_nll = discretized_nll(x_0.unsqueeze(1), p_mu, p_var)
                # loss_vlb = th.where(th.eq(t, 0), loss_nll, loss_kl)

                loss = loss_kl + loss_mse
                loss = loss.mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                denoiser_ema.update()

                grad_norm = denoiser.grad_norm()

                losses.add_result(loss)
                mse_losses.add_result(loss_mse)
                # vlb_losses.add_result(loss_vlb)
                kl_losses.add_result(loss_kl)
                # nll_losses.add_result(loss_nll)
                grad_norms.add_result(grad_norm)

                mlflow.log_metrics(
                    {
                        "loss": losses.get_last_metric(),
                        # "loss_vlb": vlb_losses.get_last_metric(),
                        "loss_kl": kl_losses.get_last_metric(),
                        # "loss_nll": nll_losses.get_last_metric(),
                        "loss_mse": mse_losses.get_last_metric(),
                        "grad_norm": grad_norms.get_last_metric(),
                    },
                    step=metric_step,
                )
                metric_step += 1

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1} - "
                    f"save {saver.curr_save} "
                    f"[{saver.curr_step} / {train_options.save_every - 1}] "
                    f"loss = {losses.get_smoothed_metric():.6f}, "
                    f"mse = {mse_losses.get_smoothed_metric():.6f}, "
                    # f"vlb = {vlb_losses.get_smoothed_metric():.6f}, "
                    f"kl = {kl_losses.get_smoothed_metric():.6f}, "
                    # f"nll = {nll_losses.get_smoothed_metric():.6f}, "
                    f"grad_norm = {grad_norms.get_smoothed_metric():.6f}"
                )

                saver.save()
