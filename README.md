# MusicDiffusion
Music synthesis with diffusion model

## Installation

First clone the project and initialize python virtual environment with `uv` :

```bash
git clone https://github.com/Ipsedo/MusicDiffusion.git
cd ./MusicDiffusion
# init / sync uv
uv sync
```

## Usage

All commands must be executed within `uv` virtual environment.

Create the dataset from audio files :

```bash
cd /path/to/MusicDiffusion
# here /path/to/music_folder contains flac music files
# /path/to/music_dataset is the folder where the tensor pickle files will be saved
python -m music_diffusion create_data "/path/to/music_folder/*.flac" "/path/to/music_dataset"
```

Run training (adapt your hyperparameters according to your choice) :

```bash
cd /path/to/MusicDiffusion
python -m music_diffusion model --cuda train your_mlflow_run_name --input-dataset /path/to/music_dataset --output-dir /path/to/train_output
```

The denoiser is conditioned on a global vector `z` computed by a small
encoder from a *reference chunk* of the same piece (another chunk than the
one being denoised), so `z` can only hold what is shared across the whole
piece : instruments, register, tempo, texture. `--z-size` sets its size
(`0` disables the conditioning) and `--cond-dropout` the probability to
drop it during training, which is needed for classifier-free guidance.

Then, when the model has converged, generate your music :

```bash
cd /path/to/MusicDiffusion
# generate 3 musics of around 10 * 4s long each with fast sample method, the whole using EMA model (10th checkpoint)
python -m music_diffusion model --cuda generate /path/to/train_output/denoiser_ema_10.pt /path/to/generated_wav_folder --ema --fast-sample 128 --frames 10 --musics 3
```

For each music, a first 4s frame is generated without condition, then
encoded into `z`, and the whole music is generated conditioned on this `z` :
the music defines its own identity and keeps it over every frame.

Note : the dataset format changed with the conditioning (chunks are saved as
`magn_phase_{song}_{chunk}.pt`), datasets created before must be regenerated
with the `create_data` command.

## References
[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Jonathan Ho, Ajay Jain, Pieter Abbeel - 2020

[2] [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710), Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, Adam Roberts - 2019

[3] [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), Alex Nichol, Prafulla Dhariwal - 2021
