from .train import TrainOptions, train


def main() -> None:
    train_options = TrainOptions(
        run_name="mnist",
        batch_size=8,
        step_batch_size=32,
        epochs=1000,
        steps=1024,
        beta_1=1e-4,
        beta_t=2e-2,
        input_channels=1,
        encoder_channels=[],
        decoder_channels=[],
        time_size=8,
        cuda=True,
        learning_rate=1e-3,
        metric_window=64,
        save_every=4096,
        output_directory="/home/samuel/PycharmProjects/MusicDiffusionModel/out/train_mnist",
        nb_samples=5,
    )

    train(train_options)


if __name__ == "__main__":
    main()
