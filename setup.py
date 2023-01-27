from setuptools import setup

setup(
    name="music_diffusion_model",
    author="Samuel Berrien",
    version="1.0",
    packages=[
        "music_diffusion_model",
        "music_diffusion_model.networks",
        "music_diffusion_model.data",
    ],
    package_data={
        "music_diffusion_model": [
            "resources/*",
        ]
    },
    url="https://github.com/Ipsedo/MusicDiffusionModel",
    license="GPL-3.0 License",
)
