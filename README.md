# MusicDiffusionModel
Music with diffusion model

Create the dataset from audio files :
```bash
$ cd /path/to/MusicDiffusionModel
$ # here /path/to/music_folder contains flac music files
$ # /path/to/music_dataset is the folder where the tensor pickle files will be saved
$ python -m music_diffusion create_data "/path/to/music_folder/*.flac" "/path/to/music_dataset"
```

Run training (adapt your hyper-parameters according to your choice) :
```bash
$ cd /path/to/MusicDiffusionModel
$ python -m music_diffusion model --cuda --unet-channels "[(8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 512)]" --use-attentions "[False, False, False, False, True, False, False]" --attention-heads 8 --time-size 64 --norm-groups 4 --steps 4096 train mlflow_run_name --batch-size 4 --step-batch-size 1 --input-dataset /path/to/music_dataset --output-dir /path/to/train_output --save-every 4096 --learning-rate 1e-4 --vlb-loss-factor 1e-3
```

# References
[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Jonathan Ho, Ajay Jain, Pieter Abbeel - 2020

[2] [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710), Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, Adam Roberts - 2019

[3] [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), Alex Nichol, Prafulla Dhariwal - 2021
