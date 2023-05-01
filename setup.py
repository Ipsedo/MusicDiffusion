# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name="music_diffusion",
    author="Samuel Berrien",
    version="1.0",
    packages=[
        "music_diffusion",
        "music_diffusion.networks",
        "music_diffusion.data",
    ],
    package_data={
        "music_diffusion": [
            "resources/*",
        ]
    },
    url="https://github.com/Ipsedo/MusicDiffusionModel",
    license="GPL-3.0 License",
)
