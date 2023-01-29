import argparse

from .data import create_dataset
from .generate import GenerateOptions, generate
from .train import TrainOptions, train


def main_generate() -> None:
    generate_options = GenerateOptions(
        steps=1024,
        beta_1=1e-4,
        beta_t=2e-2,
        input_channels=2,
        encoder_channels=[
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
            (80, 96),
        ],
        decoder_channels=[
            (96, 80),
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16),
        ],
        time_size=8,
        cuda=False,
        denoiser_dict_state="/home/samuel/PycharmProjects/MusicDiffusionModel/out/train_korn/denoiser_0.pt",
        output_dir="/home/samuel/PycharmProjects/MusicDiffusionModel/out/generate_train_korn_0",
        frames=3,
        musics=3,
    )

    generate(generate_options)


def main_create_dataset() -> None:
    create_dataset(
        "/home/samuel/Musique/Korn_all_flac/*.flac",
        "/home/samuel/PycharmProjects/MusicDiffusionModel/res/korn_dataset",
    )


def main_train() -> None:
    train_options = TrainOptions(
        run_name="korn",
        dataset_path="/home/samuel/PycharmProjects/MusicDiffusionModel/res/korn_dataset",
        batch_size=4,
        step_batch_size=1,
        epochs=1000,
        steps=1024,
        beta_1=1e-4,
        beta_t=2e-2,
        input_channels=2,
        encoder_channels=[
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
            (80, 96),
        ],
        decoder_channels=[
            (96, 80),
            (80, 64),
            (64, 48),
            (48, 32),
            (32, 16),
        ],
        time_size=8,
        cuda=True,
        learning_rate=1e-3,
        metric_window=64,
        save_every=4096,
        output_directory="/home/samuel/PycharmProjects/MusicDiffusionModel/out/train_korn",
        nb_samples=5,
    )

    train(train_options)


def main() -> None:
    parser = argparse.ArgumentParser("music_diffusion_model")

    _ = parser.parse_args()

    main_train()


if __name__ == "__main__":
    main()
