# -*- coding: utf-8 -*-
from statistics import mean

import mlflow
import torch as th
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AudioDataset
from .networks import log_likelihood, mse, normal_kl_div
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
        if train_options.optim_state_dict is not None:
            optim.load_state_dict(th.load(train_options.optim_state_dict))

        saver = Saver(
            model_options.input_channels,
            noiser,
            denoiser,
            optim,
            denoiser_ema,
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
                "norm_groups": model_options.norm_groups,
                "input_dataset": train_options.dataset_path,
            }
        )

        device = "cuda" if model_options.cuda else "cpu"

        losses = [1.0 for _ in range(train_options.metric_window)]
        mse_losses = [1.0 for _ in range(train_options.metric_window)]
        vlb_losses = [1.0 for _ in range(train_options.metric_window)]
        kl_losses = [1.0 for _ in range(train_options.metric_window)]
        ll_losses = [1.0 for _ in range(train_options.metric_window)]
        grad_norms = [1.0 for _ in range(train_options.metric_window)]
        metric_step = 0

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for x_0 in tqdm_bar:

                if model_options.cuda:
                    x_0 = x_0.cuda()

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

                loss_mse = mse(eps, eps_theta)

                q_mu, q_var = noiser.posterior(x_t, x_0, t)
                p_mu, p_var = denoiser.prior(
                    x_t, t, eps_theta.detach(), v_theta
                )

                loss_kl = normal_kl_div(q_mu, q_var, p_mu, p_var)
                loss_ll = log_likelihood(x_0, p_mu, p_var)
                loss_vlb = th.where(t == 0, loss_ll, loss_kl)

                loss = loss_vlb * 1e-3 + loss_mse
                loss = loss.mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                denoiser_ema.update()

                grad_norm = denoiser.grad_norm()

                del losses[0]
                del mse_losses[0]
                del vlb_losses[0]
                del kl_losses[0]
                del ll_losses[0]
                losses.append(loss.item())
                vlb_losses.append(loss_vlb.mean().item())
                kl_losses.append(loss_vlb.mean().item())
                ll_losses.append(loss_vlb.mean().item())
                mse_losses.append(loss_mse.mean().item())

                del grad_norms[0]
                grad_norms.append(grad_norm)

                mlflow.log_metrics(
                    {
                        "loss": losses[-1],
                        "loss_vlb": vlb_losses[-1],
                        "loss_kl": kl_losses[-1],
                        "loss_ll": ll_losses[-1],
                        "loss_mse": mse_losses[-1],
                        "grad_norm": grad_norms[-1],
                    },
                    step=metric_step,
                )
                metric_step += 1

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1} - "
                    f"save {saver.curr_save} "
                    f"[{saver.curr_step} / {train_options.save_every - 1}] "
                    f"loss = {mean(losses):.6f}, "
                    f"mse = {mean(mse_losses):.6f}, "
                    f"vlb = {mean(vlb_losses):.6f}, "
                    f"kl = {mean(kl_losses):.6f}, "
                    f"ll = {mean(ll_losses):.6f}, "
                    f"grad_norm = {mean(grad_norms):.6f}"
                )

                saver.save()
