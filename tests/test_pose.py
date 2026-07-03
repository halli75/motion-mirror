import sys
import tempfile
import types
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


# ── real-path tracking tests (fake rtmlib tracker, no GPU / no models) ───────

def _real_cfg(tmp_path: Path) -> MotionMirrorConfig:
    """Config on the real (non-mock) path with dummy DWPose model files."""
    cfg = MotionMirrorConfig(
        backend="wan-1.3b-vace", device="cpu", cache_dir=tmp_path / "cache"
    )
    model_dir = cfg.model_cache("dwpose")
    (model_dir / "dw-ll_ucoco_384.onnx").touch()
    (model_dir / "yolox_l.onnx").touch()
    return cfg


def _install_fake_rtmlib(monkeypatch, per_frame_results):
    """Insert a fake rtmlib whose Wholebody returns one queued result per call."""
    calls = {"i": 0}

    class FakeWholebody:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, frame):
            result = per_frame_results[calls["i"]]
            calls["i"] += 1
            return result

    fake = types.ModuleType("rtmlib")
    fake.Wholebody = FakeWholebody
    monkeypatch.setitem(sys.modules, "rtmlib", fake)


def _person(cx: float, cy: float, spread: float = 30.0):
    """(133, 2) xy keypoints clustered around (cx, cy) plus (133,) confidences."""
    rng = np.random.default_rng(int(cx) * 1000 + int(cy))
    xy = (rng.uniform(-spread, spread, (133, 2)) + [cx, cy]).astype(np.float32)
    scores = np.full((133,), 0.9, dtype=np.float32)
    return xy, scores


def test_pose_zero_detection_midframe_yields_zeros(tmp_path, monkeypatch):
    vid = _make_video(tmp_path / "motion.mp4", frames=3)
    cfg = _real_cfg(tmp_path)
    a0_xy, a0_sc = _person(60, 60)
    a2_xy, a2_sc = _person(64, 62)
    _install_fake_rtmlib(monkeypatch, [
        (a0_xy[np.newaxis], a0_sc[np.newaxis]),
        (np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)),
        (a2_xy[np.newaxis], a2_sc[np.newaxis]),
    ])
    result = extract_pose(vid, cfg)
    assert result.keypoints.shape == (3, 133, 3)
    assert np.all(result.keypoints[1] == 0.0)
    assert np.all(result.keypoints[1][:, 2] == 0.0)


def test_pose_tracks_nearest_person_when_order_swaps(tmp_path, monkeypatch):
    vid = _make_video(tmp_path / "motion.mp4", frames=2)
    cfg = _real_cfg(tmp_path)
    a0_xy, a0_sc = _person(40, 40)     # person A, alone on frame 0
    a1_xy, a1_sc = _person(44, 42)     # A moved slightly, listed SECOND on frame 1
    b1_xy, b1_sc = _person(100, 100)   # person B, listed FIRST on frame 1
    _install_fake_rtmlib(monkeypatch, [
        (a0_xy[np.newaxis], a0_sc[np.newaxis]),
        (np.stack([b1_xy, a1_xy]), np.stack([b1_sc, a1_sc])),
    ])
    result = extract_pose(vid, cfg)
    kept = result.keypoints[1]
    centroid = kept[kept[:, 2] > 0.3, :2].mean(axis=0)
    assert np.linalg.norm(centroid - np.array([44.0, 42.0])) < 10.0
