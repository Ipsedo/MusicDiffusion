# -*- coding: utf-8 -*-
import argparse
import json
import re
from typing import List, Tuple

from .data import create_dataset, create_metadata_csv
from .generate import generate
from .options import GenerateOptions, ModelOptions, TrainOptions
from .train import train


def _channels(string: str) -> List[Tuple[int, int]]:
    regex_match = re.compile(
        r"^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$"
    )
    regex_layer = re.compile(r"\( *\d+ *, *\d+ *\)")
    regex_channel = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [(10, 20), (20, 40), ...]"

    def _match_channels(layer_str: str) -> Tuple[int, int]:
        matched = regex_channel.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_channels(layer) for layer in regex_layer.findall(string)]


def _attentions(string: str) -> List[bool]:
    regex_true_false = re.compile(r"(?:True)|(?:False)")
    regex_match = re.compile(
        r"^ *\[(?: *(?:(?:True)|(?:False)) *,)* *(?:(?:True)|(?:False)) *] *$"
    )

    assert regex_match.match(string), "usage : [True, False, True, ...]"

    return [use_att == "True" for use_att in regex_true_false.findall(string)]


def main() -> None:
    parser = argparse.ArgumentParser("music_diffusion")

    sub_command = parser.add_subparsers(
        title="mode", dest="mode", required=True
    )

    #######################
    # Create Bach metadata
    #######################

    metadata_parser = sub_command.add_parser("metadata")

    metadata_parser.add_argument("bach_flac_folder", type=str)
    metadata_parser.add_argument("output_csv_path", type=str)

    #################
    # Create dataset
    #################

    dataset_parser = sub_command.add_parser("create_data")

    dataset_parser.add_argument("metadata_csv_path", type=str)
    dataset_parser.add_argument("output_dir", type=str)

    #####################
    # Train and Generate
    #####################

    # Model hyper parameters
    model_parser = sub_command.add_parser("model")

    model_parser.add_argument("--steps", type=int, default=4096)
    model_parser.add_argument(
        "--unet-channels",
        type=_channels,
        default=[
            (2, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 256),
        ],
    )
    model_parser.add_argument("--time-size", type=int, default=16)
    model_parser.add_argument("--tau-hidden-dim", type=int, default=64)
    model_parser.add_argument("--tau-layers", type=int, default=3)
    model_parser.add_argument("--cuda", action="store_true")

    model_parser.add_argument("--key2idx-json", type=str, required=True)
    model_parser.add_argument("--genre2idx-json", type=str, required=True)
    model_parser.add_argument("--scoring2idx-json", type=str, required=True)

    # Sub command run {train, generate}
    model_sub_command = model_parser.add_subparsers(
        title="run", dest="run", required=True
    )

    # Train parser
    train_parser = model_sub_command.add_parser("train")

    train_parser.add_argument("run_name", type=str)

    train_parser.add_argument("-i", "--input-dataset", type=str, required=True)
    train_parser.add_argument("--batch-size", type=int, default=6)
    train_parser.add_argument("--step-batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--metric-window", type=int, default=64)
    train_parser.add_argument("--save-every", type=int, default=4096)
    train_parser.add_argument("-o", "--output-dir", type=str, required=True)
    train_parser.add_argument("--nb-samples", type=int, default=5)
    train_parser.add_argument("--denoiser-state-dict", type=str)
    train_parser.add_argument("--ema-state-dict", type=str)
    train_parser.add_argument("--noiser-state-dict", type=str)
    train_parser.add_argument("--optim-state-dict", type=str)

    # Generate parser
    generate_parser = model_sub_command.add_parser("generate")

    generate_parser.add_argument("denoiser_dict_state", type=str)
    generate_parser.add_argument("output_dir", type=str)
    generate_parser.add_argument("--fast-sample", type=int, required=False)
    generate_parser.add_argument("--frames", type=int, required=True)
    generate_parser.add_argument("--musics", type=int, required=True)
    generate_parser.add_argument("--ema", action="store_true")
    generate_parser.add_argument("--magn-scale", type=float, default=1.0)

    generate_parser.add_argument("--keys", type=str, nargs="+", required=True)
    generate_parser.add_argument(
        "--genres", type=str, nargs="+", required=True
    )
    generate_parser.add_argument(
        "--scoring", type=str, nargs="+", required=True
    )

    #######
    # Main
    #######

    args = parser.parse_args()

    if args.mode == "model":

        with (
            open(args.key2idx_json, "r", encoding="utf-8") as k_f,
            open(args.genre2idx_json, "r", encoding="utf-8") as g_f,
            open(args.scoring2idx_json, "r", encoding="utf-8") as s_f,
        ):
            key_to_idx = json.load(k_f)
            genre_to_idx = json.load(g_f)
            scoring_to_idx = json.load(s_f)

        model_options = ModelOptions(
            steps=args.steps,
            unet_channels=args.unet_channels,
            time_size=args.time_size,
            tau_dim=len(key_to_idx) + len(genre_to_idx) + len(scoring_to_idx),
            tau_hidden_dim=args.tau_hidden_dim,
            tau_layers=args.tau_layers,
            cuda=args.cuda,
        )

        if args.run == "train":
            train_options = TrainOptions(
                run_name=args.run_name,
                dataset_path=args.input_dataset,
                batch_size=args.batch_size,
                step_batch_size=args.step_batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                metric_window=args.metric_window,
                save_every=args.save_every,
                output_directory=args.output_dir,
                nb_samples=args.nb_samples,
                noiser_state_dict=args.noiser_state_dict,
                denoiser_state_dict=args.denoiser_state_dict,
                ema_state_dict=args.ema_state_dict,
                optim_state_dict=args.optim_state_dict,
            )

            train(model_options, train_options)

        elif args.run == "generate":
            keys = args.keys
            genres = args.genres

            assert all(k in key_to_idx for k in keys)
            assert all(g in genre_to_idx for g in genres)

            regex_scoring = re.compile(r"^(([^ ]+ )+)?[^ ]+$")

            scoring_list = []
            for s_l in args.scoring:
                assert regex_scoring.match(
                    s_l
                ), f'Invalid scoring list : "{s_l}"'

                scoring = []
                for s in s_l.split(" "):
                    s = s.strip()
                    assert s in scoring_to_idx, f'Invalid scoring : "{s}"'
                    scoring.append(s)

                scoring_list.append(scoring)

            generate_options = GenerateOptions(
                fast_sample=args.fast_sample,
                denoiser_dict_state=args.denoiser_dict_state,
                ema_denoiser=args.ema,
                output_dir=args.output_dir,
                frames=args.frames,
                musics=args.musics,
                magn_scale=args.magn_scale,
                keys=keys,
                genres=genres,
                scoring_list=scoring_list,
                key_to_idx=key_to_idx,
                genres_to_idx=genre_to_idx,
                scoring_to_idx=scoring_to_idx,
            )

            generate(model_options, generate_options)

        else:
            parser.error(f"Unrecognized run '{args.run}'")

    elif args.mode == "create_data":
        create_dataset(
            args.metadata_csv_path,
            args.output_dir,
        )

    elif args.mode == "metadata":
        create_metadata_csv(
            args.bach_flac_folder,
            args.output_csv_path,
        )

    else:
        parser.error(f"Unrecognized mode '{args.mode}'")


if __name__ == "__main__":
    main()
