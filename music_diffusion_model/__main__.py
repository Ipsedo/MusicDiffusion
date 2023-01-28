from .train import TrainOptions, train


def main() -> None:
    train_options = TrainOptions(
        run_name="mnist",
        batch_size=8,
        epochs=1000,
        steps=256,
        beta_1=1e-4,
        beta_t=2e-2,
        input_channels=1,
        time_size=8,
        cuda=True,
        learning_rate=1e-4,
    )

    train(train_options)


if __name__ == "__main__":
    main()
