# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="music_diffusion",
    author="Samuel Berrien",
    version="1.1",
    packages=find_packages(include=["music_diffusion", "music_diffusion.*"]),
    url="https://github.com/Ipsedo/MusicDiffusionModel",
    license="GPL-3.0 License",
)
