#!/usr/bin/env bash

python -m music_diffusion model --cuda --unet-channels "[(2,32),(32,64),(64,128),(128,256),(256,512),(512,1024)]" --time-size 16 --steps 4096 generate ./denoiser_ema_100.pt ../out/out_bach_100 --frames 4 --musics 4 --ema --fast-sample 128 --magn-scale 1.0