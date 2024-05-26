# MusicDiffusionModel
Music with diffusion model

Create the dataset from audio files :
```bash
$ cd /path/to/MusicDiffusionModel
$ # here /path/to/music_folder contains flac music files
$ # /path/to/music_dataset is the folder where the tensor pickle files will be saved
$ python -m music_diffusion create_waveform "/path/to/music_folder/*.flac" "/path/to/music_dataset"
```

Run training (adapt the hyper-parameters according to your choice) :
```bash
$ cd /path/to/MusicDiffusionModel
$ python -m music_diffusion model --cuda --unet-channels "[(2,16),(16,32),(32,48),(48,64),(64,80),(80,96),(96,112),(112,128)]" --time-size 16 --steps 4096 --liquid-neurons 64 train elec_gems_normal --batch-size 10 --step-batch-size 1 --input-dataset /path/to/music_dataset --output-dir /path/to/train_output --save-every 1024 --learning-rate 1e-4 --nb-sample 4
```

# References
[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Jonathan Ho, Ajay Jain, Pieter Abbeel - 2020

[2] [GANSynth: Adversarial Neural Audio Synthesis](https://arxiv.org/abs/1902.08710), Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, Adam Roberts - 2019

[3] [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), Alex Nichol, Prafulla Dhariwal - 2021
