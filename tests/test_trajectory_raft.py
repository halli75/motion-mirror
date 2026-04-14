"""Tests for RAFT optical flow integration in trajectory synthesis (Phase B).

Non-GPU tests verify the dispatcher logic and Farneback fallback.
GPU tests exercise actual RAFT inference — require torchvision + CUDA.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)) -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 200, (h, w, 3), dtype=np.uint8))
    writer.release()
    return path


def _make_frames(count: int = 3, size: tuple[int, int] = (64, 64)) -> list[np.ndarray]:
    """Create synthetic BGR frames for direct helper testing."""
    h, w = size
    rng = np.random.default_rng(42)
    return [rng.integers(0, 200, (h, w, 3), dtype=np.uint8) for _ in range(count)]


# ── _compute_flow_farneback ───────────────────────────────────────────────────

def test_farneback_returns_correct_shape():
    from motion_mirror.extract.trajectory import _compute_flow_farneback
    frames = _make_frames(2, (32, 48))
    flow = _compute_flow_farneback(frames[0], frames[1])
    assert flow.shape == (32, 48, 2)
    assert flow.dtype == np.float32


def test_farneback_identical_frames_near_zero_flow():
    from motion_mirror.extract.trajectory import _compute_flow_farneback
    frames = _make_frames(1, (32, 32))
    flow = _compute_flow_farneback(frames[0], frames[0])
    assert np.abs(flow).max() < 0.5  # identical frames → near-zero flow


# ── _compute_flow_pair dispatcher ─────────────────────────────────────────────

def test_flow_pair_farneback_dispatch():
    from motion_mirror.extract.trajectory import _compute_flow_pair
    frames = _make_frames(2, (32, 32))
    flow = _compute_flow_pair(frames[0], frames[1], estimator="farneback")
    assert flow.shape == (32, 32, 2)
    assert flow.dtype == np.float32


def test_flow_pair_raft_falls_back_when_unavailable():
    """If RAFT/torchvision is not installed, should warn and fall back to Farneback."""
    from motion_mirror.extract.trajectory import _compute_flow_pair
    frames = _make_frames(2, (32, 32))

    # Simulate RAFT raising ImportError
    with patch(
        "motion_mirror.extract.trajectory._compute_flow_raft",
        side_effect=ImportError("torchvision not installed"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            flow = _compute_flow_pair(frames[0], frames[1], estimator="raft")

    assert flow.shape == (32, 32, 2)
    assert any("farneback" in str(w.message).lower() or "raft" in str(w.message).lower()
               for w in caught), "Expected fallback warning"


def test_flow_pair_raft_falls_back_on_runtime_error():
    """A GPU runtime error during RAFT should also trigger Farneback fallback."""
    from motion_mirror.extract.trajectory import _compute_flow_pair
    frames = _make_frames(2, (32, 32))

    with patch(
        "motion_mirror.extract.trajectory._compute_flow_raft",
        side_effect=RuntimeError("CUDA OOM"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            flow = _compute_flow_pair(frames[0], frames[1], estimator="raft")

    assert flow.shape == (32, 32, 2)
    assert len(caught) >= 1


# ── synthesize_trajectory with flow_estimator="raft" (mocked RAFT) ───────────

def test_synthesize_trajectory_raft_config_threads_through(tmp_path):
    """flow_estimator='raft' should reach _compute_flow_pair with estimator='raft'."""
    from PIL import Image
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.types import PoseSequence, SegmentationResult
    from motion_mirror.extract.trajectory import synthesize_trajectory

    vid = _make_video(tmp_path / "m.mp4", frames=4)

    # Build synthetic inputs
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:54] = 255
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, 3] = mask
    img_path = tmp_path / "char.png"
    Image.fromarray(rgba, "RGBA").save(str(img_path))

    seg = SegmentationResult(
        source_image_path=img_path,
        rgba_path=img_path,
        mask=mask,
        rgba=rgba,
    )

    rng = np.random.default_rng(0)
    kps = rng.random((4, 133, 3)).astype(np.float32)
    kps[:, :, 2] = kps[:, :, 2] * 0.5 + 0.5
    kps[:, :, 0] *= 64
    kps[:, :, 1] *= 64
    pose = PoseSequence(source_video_path=vid, keypoints=kps, frame_size=(64, 64), fps=24.0)

    cfg = MotionMirrorConfig(
        backend="mock", device="cpu",
        flow_estimator="raft", trajectory_density=32,
        project_root=tmp_path,
    )

    called_with_raft = []

    original_compute = None
    from motion_mirror.extract import trajectory as traj_module

    def fake_compute_flow_pair(f0, fk, estimator="farneback", device="cpu"):
        called_with_raft.append(estimator)
        # Delegate to farneback regardless so the test stays CPU-only
        from motion_mirror.extract.trajectory import _compute_flow_farneback
        return _compute_flow_farneback(f0, fk)

    with patch.object(traj_module, "_compute_flow_pair", side_effect=fake_compute_flow_pair):
        result = synthesize_trajectory(pose, seg, vid, cfg)

    assert result.tracks.shape[1] == 32, "density mismatch"
    assert any(e == "raft" for e in called_with_raft), (
        f"Expected 'raft' in calls but got {called_with_raft}"
    )


# ── GPU tests ─────────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_compute_flow_raft_real(tmp_path):
    """Real RAFT inference — requires torchvision + CUDA."""
    from motion_mirror.extract.trajectory import _compute_flow_raft
    # Clear cache so fresh model loads
    import motion_mirror.extract.trajectory as t
    t._raft_cache.clear()

    frames = _make_frames(2, (128, 128))
    flow = _compute_flow_raft(frames[0], frames[1], device="cuda")
    assert flow.shape == (128, 128, 2)
    assert flow.dtype == np.float32
    t._raft_cache.clear()


@pytest.mark.gpu
def test_flow_pair_raft_gpu_end_to_end():
    """_compute_flow_pair with estimator='raft' uses GPU RAFT on a CUDA machine."""
    from motion_mirror.extract.trajectory import _compute_flow_pair, _raft_cache
    _raft_cache.clear()
    frames = _make_frames(2, (64, 64))
    flow = _compute_flow_pair(frames[0], frames[1], estimator="raft", device="cuda")
    assert flow.shape == (64, 64, 2)
    _raft_cache.clear()
