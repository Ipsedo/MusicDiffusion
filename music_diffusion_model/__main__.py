from .data import create_dataset
from .train import TrainOptions, train


def main_create_dataset() -> None:
    create_dataset(
        "/home/samuel/Musique/Korn_all_flac/*.flac",
        "/home/samuel/PycharmProjects/MusicDiffusionModel/res/korn_dataset",
    )


def main() -> None:
    train_options = TrainOptions(
        run_name="korn",
        dataset_path="/home/samuel/PycharmProjects/MusicDiffusionModel/res/korn_dataset",
        batch_size=4,
        step_batch_size=2,
        epochs=1000,
        steps=1024,
        beta_1=1e-4,
        beta_t=2e-2,
        input_channels=2,
        encoder_channels=[
            (8, 16),
            (16, 24),
            (24, 32),
            (32, 40),
            (40, 48),
            (48, 56),
            (56, 64),
        ],
        decoder_channels=[
            (64, 56),
            (56, 48),
            (48, 40),
            (40, 32),
            (32, 24),
            (24, 16),
            (16, 8),
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


if __name__ == "__main__":
    main()
