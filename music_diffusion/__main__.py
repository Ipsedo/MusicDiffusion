import argparse
import re
from typing import List, Tuple

from .data import create_dataset
from .generate import GenerateOptions, generate
from .train import TrainOptions, train
from .utils import ModelOptions


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

    #################
    # Create dataset
    #################

    dataset_parser = sub_command.add_parser("create_data")

    dataset_parser.add_argument("music_glob_path", type=str)
    dataset_parser.add_argument("output_dir", type=str)

    #####################
    # Train and Generate
    #####################

    # Model hyper parameters
    model_parser = sub_command.add_parser("model")

    model_parser.add_argument("--steps", type=int, default=1024)
    model_parser.add_argument("--beta-1", type=float, default=1e-4)
    model_parser.add_argument("--beta-t", type=float, default=2e-2)
    model_parser.add_argument("--channels", type=int, default=2)
    model_parser.add_argument("--norm-groups", type=int, default=4)
    model_parser.add_argument(
        "--unet-channels",
        type=_channels,
        default=[
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 512),
        ],
    )
    model_parser.add_argument(
        "--use-attentions",
        type=_attentions,
        default=[
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ],
    )
    model_parser.add_argument("--attention-heads", type=int, default=8)
    model_parser.add_argument("--time-size", type=int, default=128)
    model_parser.add_argument("--cuda", action="store_true")

    # Sub command run {train, generate}
    model_sub_command = model_parser.add_subparsers(
        title="run", dest="run", required=True
    )

    # Train parser
    train_parser = model_sub_command.add_parser("train")

    train_parser.add_argument("run_name", type=str)

    train_parser.add_argument("-i", "--input-dataset", type=str, required=True)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--step-batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--metric-window", type=int, default=64)
    train_parser.add_argument("--save-every", type=int, default=4096)
    train_parser.add_argument("-o", "--output-dir", type=str, required=True)
    train_parser.add_argument("--nb-samples", type=int, default=5)
    train_parser.add_argument("--vlb-loss-factor", type=float, default=1e-3)

    # Generate parser
    generate_parser = model_sub_command.add_parser("generate")

    generate_parser.add_argument("denoiser_dict_state", type=str)
    generate_parser.add_argument("output_dir", type=str)
    generate_parser.add_argument("--frames", type=int, required=True)
    generate_parser.add_argument("--musics", type=int, required=True)

    #######
    # Main
    #######

    args = parser.parse_args()

    if args.mode == "model":
        model_options = ModelOptions(
            steps=args.steps,
            beta_1=args.beta_1,
            beta_t=args.beta_t,
            input_channels=args.channels,
            norm_groups=args.norm_groups,
            unet_channels=args.unet_channels,
            use_attentions=args.use_attentions,
            attention_heads=args.attention_heads,
            time_size=args.time_size,
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
                vlb_loss_factor=args.vlb_loss_factor,
                noiser_state_dict=None,
                denoiser_state_dict=None,
                optim_state_dict=None,
            )

            train(model_options, train_options)

        elif args.run == "generate":
            generate_options = GenerateOptions(
                denoiser_dict_state=args.denoiser_dict_state,
                output_dir=args.output_dir,
                frames=args.frames,
                musics=args.musics,
            )

            generate(model_options, generate_options)

        else:
            parser.error(f"Unrecognized run '{args.run}'")

    elif args.mode == "create_data":
        create_dataset(
            args.music_glob_path,
            args.output_dir,
        )

    else:
        parser.error(f"Unrecognized mode '{args.mode}'")


if __name__ == "__main__":
    main()
