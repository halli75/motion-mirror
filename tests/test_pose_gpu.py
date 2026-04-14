"""GPU tests for the real DWPose inference path.

Requires:
- A CUDA GPU
- DWPose ONNX weights downloaded: motion-mirror download --model dwpose
- pip install rtmlib onnxruntime-gpu

Run with:
    pytest -m gpu tests/test_pose_gpu.py -v

Note on synthetic video: YOLOX may detect zero people or a very small
false-positive in random noise. NoPoseDetectedError and SmallSubjectError
are correct behaviours in that case — the tests skip rather than fail,
because the goal is to validate that the real inference path runs without
crashing, not to assert correctness on meaningless input.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2
from pathlib import Path

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.exceptions import NoPoseDetectedError, SmallSubjectError
from motion_mirror.extract.pose import extract_pose


def _make_video(path: Path, frames: int = 5, size: tuple[int, int] = (256, 256)) -> Path:
    """Larger frame so any YOLOX false-positive is more likely to meet the 5% threshold."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 200, (h, w, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.gpu
def test_extract_pose_real_shape(tmp_path):
    """Real DWPose inference — output shape must be (F, 133, 3) when a person is found."""
    vid_path = _make_video(tmp_path / "motion.mp4", frames=5)
    cfg = MotionMirrorConfig(backend="wan-move-14b", device="cuda")
    try:
        pose = extract_pose(vid_path, cfg)
    except (NoPoseDetectedError, SmallSubjectError) as exc:
        pytest.skip(f"No valid person in synthetic noise video (expected): {exc}")

    assert pose.keypoints.shape == (5, 133, 3), (
        f"Expected (5, 133, 3), got {pose.keypoints.shape}"
    )
    assert pose.fps > 0, "FPS must be positive"
    assert pose.frame_size == (256, 256), f"Expected (256, 256), got {pose.frame_size}"


@pytest.mark.gpu
def test_extract_pose_real_confidence_range(tmp_path):
    """Confidence values (keypoints[:, :, 2]) must be in [0, 1] when a person is found."""
    vid_path = _make_video(tmp_path / "motion.mp4", frames=3)
    cfg = MotionMirrorConfig(backend="wan-move-14b", device="cuda")
    try:
        pose = extract_pose(vid_path, cfg)
    except (NoPoseDetectedError, SmallSubjectError) as exc:
        pytest.skip(f"No valid person in synthetic noise video (expected): {exc}")

    conf = pose.keypoints[:, :, 2]
    assert conf.min() >= 0.0, f"Confidence below 0: {conf.min()}"
    assert conf.max() <= 1.0, f"Confidence above 1: {conf.max()}"


@pytest.mark.gpu
def test_extract_pose_real_missing_weights_raises(tmp_path):
    """FileNotFoundError raised when DWPose ONNX models are not present."""
    vid_path = _make_video(tmp_path / "motion.mp4", frames=3)
    cfg = MotionMirrorConfig(
        backend="wan-move-14b",
        device="cuda",
        cache_dir=tmp_path / "empty_cache",
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        extract_pose(vid_path, cfg)
