from os.path import exists, isfile
from pathlib import Path

import pytest
import torch as th
from ema_pytorch import EMA

from music_diffusion.generate import generate
from music_diffusion.options import GenerateOptions, ModelOptions


@pytest.fixture(name="model_options", scope="module")
def get_model_options() -> ModelOptions:
    return ModelOptions(
        steps=4,
        unet_channels=[(2, 4)],
        unet_group_norm_nums=[2],
        time_size=2,
        z_size=3,
        cuda=False,
    )


@pytest.fixture(name="state_dicts", scope="module")
def get_state_dicts(
    tmp_path_factory: pytest.TempPathFactory, model_options: ModelOptions
) -> tuple[str, str]:
    out_dir = tmp_path_factory.mktemp("states")

    denoiser = model_options.new_denoiser()
    ema = EMA(denoiser, include_online_model=True)

    denoiser_path = out_dir / "denoiser_0.pt"
    ema_path = out_dir / "denoiser_ema_0.pt"

    th.save(denoiser.state_dict(), denoiser_path)
    th.save(ema.state_dict(), ema_path)

    return str(denoiser_path), str(ema_path)


def _generate_options(
    state_dict: str,
    ema: bool,
    output_dir: Path,
    fast_sample: int | None = 2,
    frames: int = 1,
    musics: int = 2,
) -> GenerateOptions:
    return GenerateOptions(
        fast_sample=fast_sample,
        denoiser_dict_state=state_dict,
        ema_denoiser=ema,
        output_dir=str(output_dir),
        frames=frames,
        musics=musics,
    )


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("fast_sample", [None, 2])
def test_generate(
    tmp_path: Path,
    model_options: ModelOptions,
    state_dicts: tuple[str, str],
    ema: bool,
    fast_sample: int | None,
) -> None:
    state_dict = state_dicts[1] if ema else state_dicts[0]
    out_dir = tmp_path / "out"

    generate(
        model_options,
        _generate_options(state_dict, ema, out_dir, fast_sample),
    )

    for i in range(2):
        assert isfile(out_dir / f"sound_{i}.wav")


@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("musics", [1, 3])
def test_generate_frames_and_musics(
    tmp_path: Path,
    model_options: ModelOptions,
    state_dicts: tuple[str, str],
    frames: int,
    musics: int,
) -> None:
    out_dir = tmp_path / "out"

    generate(
        model_options,
        _generate_options(
            state_dicts[0], False, out_dir, frames=frames, musics=musics
        ),
    )

    files = sorted(f for f in out_dir.iterdir() if f.suffix == ".wav")
    assert [f.name for f in files] == [f"sound_{i}.wav" for i in range(musics)]


def test_generate_without_z(tmp_path: Path) -> None:
    model_options = ModelOptions(
        steps=4,
        unet_channels=[(2, 4)],
        unet_group_norm_nums=[2],
        time_size=2,
        z_size=0,
        cuda=False,
    )

    state_dir = tmp_path / "states"
    state_dir.mkdir()
    state_dict = state_dir / "denoiser_0.pt"
    th.save(model_options.new_denoiser().state_dict(), state_dict)

    out_dir = tmp_path / "out"

    generate(model_options, _generate_options(str(state_dict), False, out_dir))

    assert isfile(out_dir / "sound_0.wav")


def test_generate_output_is_file(
    tmp_path: Path, model_options: ModelOptions, state_dicts: tuple[str, str]
) -> None:
    out_file = tmp_path / "out"
    out_file.write_text("not a directory")

    assert exists(out_file)

    with pytest.raises(NotADirectoryError):
        generate(
            model_options,
            _generate_options(state_dicts[0], False, out_file),
        )
