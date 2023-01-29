import argparse
import re
from typing import List, Tuple

from .data import create_dataset
from .generate import GenerateOptions, generate
from .train import TrainOptions, train


def _channels(string: str) -> List[Tuple[int, int]]:
    regex_layer = re.compile(r"(\( *\d+ *, *\d+ *\))")
    regex_channel = re.compile(r"\d+")

    def _match_channels(layer_str: str) -> Tuple[int, int]:
        matched = regex_channel.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_channels(layer) for layer in regex_layer.findall(string)]


def main() -> None:
    parser = argparse.ArgumentParser("music_diffusion_model")

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
    model_parser.add_argument(
        "--encoder-channels",
        type=_channels,
        default=[
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
            (80, 96),
        ],
    )
    model_parser.add_argument(
        "--decoder-channels",
        type=_channels,
        default=[
            (96, 80),
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16),
        ],
    )
    model_parser.add_argument("--time-size", type=int, default=8)
    model_parser.add_argument("--cuda", action="store_true")

    # Sub command run {train, generate}
    model_sub_command = model_parser.add_subparsers(
        title="run", dest="run", required=True
    )

    # Train parser
    train_parser = model_sub_command.add_parser("train")

    train_parser.add_argument("run_name", type=str)

    train_parser.add_argument("-i", "--input-dataset", type=str, required=True)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--step-batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--metric-window", type=int, default=64)
    train_parser.add_argument("--save-every", type=int, default=4096)
    train_parser.add_argument("-o", "--output-dir", type=str, required=True)
    train_parser.add_argument("--nb-samples", type=int, default=5)

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

    match args.mode:
        case "model":

            match args.run:
                case "train":
                    train_options = TrainOptions(
                        run_name=args.run_name,
                        dataset_path=args.input_dataset,
                        batch_size=args.batch_size,
                        step_batch_size=args.step_batch_size,
                        epochs=args.epochs,
                        steps=args.steps,
                        beta_1=args.beta_1,
                        beta_t=args.beta_t,
                        input_channels=args.channels,
                        encoder_channels=args.encoder_channels,
                        decoder_channels=args.decoder_channels,
                        time_size=args.time_size,
                        cuda=args.cuda,
                        learning_rate=args.learning_rate,
                        metric_window=args.metric_window,
                        save_every=args.save_every,
                        output_directory=args.output_dir,
                        nb_samples=args.nb_samples,
                    )

                    train(train_options)

                case "generate":
                    generate_options = GenerateOptions(
                        steps=args.steps,
                        beta_1=args.beta_1,
                        beta_t=args.beta_t,
                        input_channels=args.input_channels,
                        encoder_channels=args.encoder_channels,
                        decoder_channels=args.decoder_channels,
                        time_size=args.time_size,
                        cuda=args.cuda,
                        denoiser_dict_state=args.denoiser_dict_state,
                        output_dir=args.output_dir,
                        frames=args.frames,
                        musics=args.musics,
                    )

                    generate(generate_options)

        case "create_data":
            create_dataset(
                args.music_glob_path,
                args.output_dir,
            )


if __name__ == "__main__":
    main()
