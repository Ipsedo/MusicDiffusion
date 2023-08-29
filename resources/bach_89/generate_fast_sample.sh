#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# fit RTX 3060 6GB memory
python -m  music_diffusion model --cuda --unet-channels "[(16,32),(32,48),(48,64),(64,80)]" --time-size 48 --norm-groups 16 --steps 4096 --beta-1 2.5e-5 --beta-t 2e-2 generate "${SCRIPT_DIR}/denoiser_89.pt" "./out_bach_89_fast-sample" --frames 4 --musics 3 --fast-sample 128