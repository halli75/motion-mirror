import tempfile
from pathlib import Path

import numpy as np
import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.extract.pose import extract_pose
from motion_mirror.types import PoseSequence


def _make_video(path: Path, frames: int = 5, size: tuple[int, int] = (128, 128)) -> Path:
    """Write a minimal synthetic MP4 using OpenCV."""
    import cv2

    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h))
    rng = np.random.default_rng(42)
    for _ in range(frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ── mock mode tests (no GPU / no models) ─────────────────────────────────────

def test_pose_returns_pose_sequence(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert isinstance(result, PoseSequence)


def test_pose_keypoints_shape(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert result.keypoints.shape == (5, 133, 3)
    assert result.keypoints.dtype == np.float32


def test_pose_frame_size(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=3, size=(160, 120))
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert result.frame_size == (160, 120)


def test_pose_fps_positive(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=3)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert result.fps > 0


def test_pose_source_path_preserved(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=3)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert result.source_video_path == vid


def test_pose_mock_confidence_range(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=4)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    conf = result.keypoints[:, :, 2]
    assert conf.min() >= 0.5
    assert conf.max() <= 1.0


# ── validation tests ──────────────────────────────────────────────────────────

def test_pose_missing_file_raises(tmp_path):
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    with pytest.raises(FileNotFoundError, match="Video not found"):
        extract_pose(tmp_path / "nonexistent.mp4", cfg)


def test_pose_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "file.wmv"
    bad.write_bytes(b"fake")
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    from motion_mirror.exceptions import UnsupportedVideoError
    with pytest.raises(UnsupportedVideoError):
        extract_pose(bad, cfg)


def test_pose_frame_count_matches_video(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=8)
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    result = extract_pose(vid, cfg)
    assert result.keypoints.shape[0] == 8
