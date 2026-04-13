"""Tests for generation stage — mock path only (no GPU required)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.controlnet import generate_with_controlnet
from motion_mirror.generate.models import GenerationRequest
from motion_mirror.generate.wan_move import generate_with_wan_move
from motion_mirror.types import GenerationResult


def _mock_request(tmp_path: Path, resolution: str = "128x64", frames: int = 4, seed: int = 0) -> GenerationRequest:
    return GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated.mp4",
        backend="mock",
        resolution=resolution,
        frames=frames,
        device="cpu",
        seed=seed,
    )


def _mock_cfg(tmp_path: Path) -> MotionMirrorConfig:
    return MotionMirrorConfig(project_root=tmp_path, backend="mock", device="cpu")


# ── wan_move mock path ────────────────────────────────────────────────────────

def test_wan_move_returns_generation_result(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert isinstance(result, GenerationResult)


def test_wan_move_output_file_exists(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert result.video_path.exists()


def test_wan_move_output_is_readable_video(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=4)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == req.frames


def test_wan_move_result_metadata(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=5)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert result.backend == "mock"
    assert result.resolution == "128x64"
    assert result.num_frames == 5


def test_wan_move_creates_output_dir(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "deep" / "nested" / "generated.mp4",
        backend="mock",
        resolution="64x64",
        frames=2,
        device="cpu",
    )
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert result.video_path.exists()


def test_wan_move_different_seeds_produce_different_colours(tmp_path):
    req0 = _mock_request(tmp_path, seed=0)
    req1 = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated_1.mp4",
        backend="mock",
        resolution="128x64",
        frames=4,
        device="cpu",
        seed=99,
    )
    cfg = _mock_cfg(tmp_path)
    r0 = generate_with_wan_move(req0, cfg)
    r1 = generate_with_wan_move(req1, cfg)

    cap0 = cv2.VideoCapture(str(r0.video_path))
    cap1 = cv2.VideoCapture(str(r1.video_path))
    _, f0 = cap0.read(); cap0.release()
    _, f1 = cap1.read(); cap1.release()
    # Mean pixel values should differ (different seed → different colour)
    assert not np.allclose(f0.mean(), f1.mean(), atol=1.0)


def test_wan_move_invalid_resolution_raises(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="mock",
        resolution="bad_resolution",
        frames=2,
        device="cpu",
    )
    cfg = _mock_cfg(tmp_path)
    with pytest.raises(ValueError, match="Invalid resolution"):
        generate_with_wan_move(req, cfg)


def test_wan_move_real_path_raises_not_implemented(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="wan-move-14b",
        resolution="128x64",
        frames=2,
        device="cpu",
    )
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="wan-move-14b", device="cpu")
    with pytest.raises((NotImplementedError, FileNotFoundError)):
        generate_with_wan_move(req, cfg)


# ── controlnet mock path ──────────────────────────────────────────────────────

def test_controlnet_returns_generation_result(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    assert isinstance(result, GenerationResult)


def test_controlnet_output_file_exists(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    assert result.video_path.exists()


def test_controlnet_output_is_readable_video(tmp_path):
    req = _mock_request(tmp_path, frames=3)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == req.frames


def test_controlnet_real_path_raises_not_implemented(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="controlnet",
        resolution="128x64",
        frames=2,
        device="cpu",
    )
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="controlnet", device="cpu")
    with pytest.raises(NotImplementedError):
        generate_with_controlnet(req, cfg)
