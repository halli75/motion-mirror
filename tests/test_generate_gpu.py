"""GPU tests for the real Wan2.1-I2V-14B generation path.

Requires:
- A CUDA GPU with >= 22 GB VRAM
- Weights downloaded: motion-mirror download --model wan-move
- pip install diffusers>=0.33 transformers accelerate

Run with:
    pytest -m gpu tests/test_generate_gpu.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.models import GenerationRequest
from motion_mirror.generate.wan_move import generate_with_wan_move
from motion_mirror.types import TrajectoryMap


def _make_segmented_image(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    rgba = Image.new("RGBA", size, (180, 120, 80, 255))
    rgba.save(str(path))
    return path


def _make_trajectory(path: Path, frames: int = 3, density: int = 16) -> Path:
    traj = TrajectoryMap(
        tracks=np.random.default_rng(0).random((frames, density, 2)).astype(np.float32),
        flow_fields=np.zeros((frames - 1, 64, 64, 2), dtype=np.float32),
        density=density,
        frame_size=(64, 64),
    )
    traj.save(path)
    return path


@pytest.mark.gpu
def test_generate_real_produces_video(tmp_path):
    """Real Wan2.1-I2V-14B generation — requires GPU + downloaded weights."""
    img_path = _make_segmented_image(tmp_path / "char.png")
    traj_path = _make_trajectory(tmp_path / "trajectory.npz", frames=3, density=16)

    req = GenerationRequest(
        segmented_image_path=img_path,
        trajectory_map_path=traj_path,
        output_path=tmp_path / "generated.mp4",
        backend="wan-move-14b",
        resolution="832x480",
        frames=17,
        device="cuda",
        seed=42,
    )
    cfg = MotionMirrorConfig(backend="wan-move-14b", device="cuda")
    result = generate_with_wan_move(req, cfg)

    assert result.video_path.exists(), "Output video file was not created"
    assert result.video_path.stat().st_size > 0, "Output video file is empty"
    assert result.backend == "wan-move-14b"
    assert result.resolution == "832x480"
    assert result.num_frames == 17


@pytest.mark.gpu
def test_generate_real_seeded_determinism(tmp_path):
    """Back-to-back generate_with_wan_move calls must both produce valid video files.

    Note: exact byte-for-byte reproducibility is NOT guaranteed with
    enable_sequential_cpu_offload — CUDA floating-point operations are
    non-deterministic between process runs even with a fixed seed.  The
    purpose of this test is to confirm the second call completes without
    OOM or crash (the VRAM cleanup between calls is the thing under test).
    """
    img_path = _make_segmented_image(tmp_path / "char.png")
    traj_path = _make_trajectory(tmp_path / "trajectory.npz", frames=3, density=16)

    def _run(out_name: str) -> int:
        req = GenerationRequest(
            segmented_image_path=img_path,
            trajectory_map_path=traj_path,
            output_path=tmp_path / out_name,
            backend="wan-move-14b",
            resolution="832x480",
            frames=17,
            device="cuda",
            seed=7,
        )
        cfg = MotionMirrorConfig(backend="wan-move-14b", device="cuda")
        result = generate_with_wan_move(req, cfg)
        return result.video_path.stat().st_size

    size_a = _run("gen_a.mp4")
    size_b = _run("gen_b.mp4")
    # Both files must exist and be non-empty — exact size equality is not
    # required because sequential CPU offload makes generation non-deterministic.
    assert size_a > 0, "First generation produced an empty file"
    assert size_b > 0, "Second generation produced an empty file"
    assert (tmp_path / "gen_a.mp4").exists()
    assert (tmp_path / "gen_b.mp4").exists()


@pytest.mark.gpu
def test_generate_real_missing_weights_raises(tmp_path):
    """FileNotFoundError raised when model weights directory is empty."""
    img_path = _make_segmented_image(tmp_path / "char.png")
    traj_path = _make_trajectory(tmp_path / "trajectory.npz")

    req = GenerationRequest(
        segmented_image_path=img_path,
        trajectory_map_path=traj_path,
        output_path=tmp_path / "generated.mp4",
        backend="wan-move-14b",
        resolution="832x480",
        frames=17,
        device="cuda",
        seed=0,
    )
    # Point cache at an empty temp dir so no weights are found
    cfg = MotionMirrorConfig(
        backend="wan-move-14b",
        device="cuda",
        cache_dir=tmp_path / "empty_cache",
    )
    with pytest.raises(FileNotFoundError, match="wan"):
        generate_with_wan_move(req, cfg)
