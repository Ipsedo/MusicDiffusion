# -*- coding: utf-8 -*-
from math import sqrt
from typing import Final, Tuple

N_FFT: Final[int] = 1024
N_VEC: Final[int] = 512
STFT_STRIDE: Final[int] = 128

SAMPLE_RATE: Final[int] = 16000

OUTPUT_SIZES: Final[Tuple[int, int]] = (N_FFT // 2, N_VEC)

N_SAMPLES: Final[int] = 2**18
N_SAMPLES_SHIFT: Final[int] = 2**16

BIN_SIZE: Final[float] = sqrt(1.0 / 128.0)
