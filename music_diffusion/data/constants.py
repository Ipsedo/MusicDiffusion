# -*- coding: utf-8 -*-
from typing import Final, Tuple

N_FFT: Final[int] = 1024
N_VEC: Final[int] = 1024
STFT_STRIDE: Final[int] = 128

SAMPLE_RATE: Final[int] = 16000

OUTPUT_SIZES: Final[Tuple[int, int]] = (N_FFT // 2, N_VEC)

BIN_SIZE: Final[float] = 1.0 / 2.0**16.0
