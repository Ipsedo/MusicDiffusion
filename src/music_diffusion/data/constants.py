from typing import Final

N_FFT: Final[int] = 1024
N_VEC: Final[int] = 512
STFT_STRIDE: Final[int] = 128

SAMPLE_RATE: Final[int] = 16000

OUTPUT_SIZES: Final[tuple[int, int]] = (N_FFT // 2, N_VEC)

BIN_SIZE: Final[float] = 1.0 / 2.0**16.0

TOP_DB: Final[float] = 80.0

# magnitude channel statistics (in [-1, 1] dB scale) over the training dataset
MAGN_MEAN: Final[float] = -0.5064
MAGN_STD: Final[float] = 0.4303

# standardized magnitude bounds
MAGN_MIN: Final[float] = (-1.0 - MAGN_MEAN) / MAGN_STD
MAGN_MAX: Final[float] = (1.0 - MAGN_MEAN) / MAGN_STD
