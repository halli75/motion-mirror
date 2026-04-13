"""Tests for trajectory synthesis (Phase 5).

All tests are CPU-only (no GPU / no model weights required).
Synthetic inputs are used throughout:
  - PoseSequence with linearly drifting keypoints
  - SegmentationResult with a solid rectangular mask
  - A synthetic MP4 written with cv2.VideoWriter
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.extract.trajectory import (
    _apply_transform_to_points,
    _build_body_transform,
    _build_nonrigid_mask,
    _layer1_skeleton_tracks,
    _layer2_interpolated_tracks,
)
from motion_mirror.extract.trajectory import synthesize_trajectory
from motion_mirror.types import PoseSequence, SegmentationResult, TrajectoryMap


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_video(path: Path, frames: int = 5, size: tuple[int, int] = (128, 128)) -> Path:
    """Write a synthetic MP4 with random BGR frames."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        frame = rng.integers(0, 200, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_drifting_pose(
    num_frames: int = 5,
    frame_size: tuple[int, int] = (128, 128),
    drift_x: float = 2.0,
) -> PoseSequence:
    """PoseSequence where body keypoints (0-16) drift +drift_x px per frame."""
    fw, fh = frame_size
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    for f in range(num_frames):
        kps[f, :17, 0] = 40.0 + f * drift_x   # x drifts right
        kps[f, :17, 1] = 50.0                  # y static
        kps[f, :17, 2] = 0.9                   # high confidence
        # Non-body keypoints: low confidence so they're ignored by Layer 1
        kps[f, 17:, 2] = 0.1
    return PoseSequence(
        source_video_path=Path("motion.mp4"),
        keypoints=kps,
        frame_size=frame_size,
        fps=24.0,
    )


def _make_segmentation(
    tmp_path: Path,
    size: tuple[int, int] = (128, 128),
) -> SegmentationResult:
    """SegmentationResult with a solid rectangular foreground mask."""
    h, w = size[1], size[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Fill centre 50% rectangle as foreground
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = mask
    rgba_path = tmp_path / "segmented.png"
    from PIL import Image
    Image.fromarray(rgba, "RGBA").save(str(rgba_path))
    return SegmentationResult(
        source_image_path=tmp_path / "char.png",
        rgba_path=rgba_path,
        mask=mask,
        rgba=rgba,
    )


def _make_config(tmp_path: Path) -> MotionMirrorConfig:
    return MotionMirrorConfig(
        project_root=tmp_path,
        backend="mock",
        device="cpu",
        trajectory_density=64,  # small for speed
    )


# ── synthesize_trajectory: output contract ────────────────────────────────────


def test_trajectory_returns_trajectory_map(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    pose = _make_drifting_pose(num_frames=5)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    assert isinstance(result, TrajectoryMap)


def test_trajectory_tracks_shape(tmp_path):
    density = 64
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    pose = _make_drifting_pose(num_frames=5)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    assert result.tracks.shape == (5, density, 2), result.tracks.shape


def test_trajectory_tracks_dtype(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    pose = _make_drifting_pose(num_frames=5)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    assert result.tracks.dtype == np.float32


def test_trajectory_values_in_unit_range(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    pose = _make_drifting_pose(num_frames=5)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    assert result.tracks.min() >= 0.0, f"min={result.tracks.min()}"
    assert result.tracks.max() <= 1.0, f"max={result.tracks.max()}"


def test_trajectory_flow_fields_shape(tmp_path):
    frames = 5
    vid = _make_video(tmp_path / "motion.mp4", frames=frames, size=(64, 64))
    pose = _make_drifting_pose(num_frames=frames, frame_size=(64, 64))
    seg = _make_segmentation(tmp_path, size=(64, 64))
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    # flow_fields: (F-1, H, W, 2)
    assert result.flow_fields.shape[0] == frames - 1
    assert result.flow_fields.shape[3] == 2


def test_trajectory_density_matches_config(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    pose = _make_drifting_pose(num_frames=5)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    cfg_128 = MotionMirrorConfig(
        project_root=tmp_path,
        backend="mock",
        device="cpu",
        trajectory_density=128,
    )
    result = synthesize_trajectory(pose, seg, vid, cfg_128)
    assert result.tracks.shape[1] == 128
    assert result.density == 128


def test_trajectory_frame_size_matches_char_image(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5, size=(128, 128))
    pose = _make_drifting_pose(num_frames=5, frame_size=(128, 128))
    seg = _make_segmentation(tmp_path, size=(64, 64))
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)
    assert result.frame_size == (64, 64)


# ── Layer 1: drift correctness ────────────────────────────────────────────────


def test_layer1_rightward_drift():
    """Layer-1 tracks must drift right when keypoints move right."""
    num_frames = 5
    char_size = (128, 128)
    # Build identity-like transform (reference frame == char image size)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    for f in range(num_frames):
        kps[f, :17, 0] = 30.0 + f * 3.0  # 3 px/frame rightward
        kps[f, :17, 1] = 60.0
        kps[f, :17, 2] = 0.95

    pose = PoseSequence(
        source_video_path=Path("v.mp4"),
        keypoints=kps,
        frame_size=char_size,
        fps=24.0,
    )
    M = _build_body_transform(kps, char_size, char_size)
    tracks = _layer1_skeleton_tracks(kps, M, char_size)

    # x coordinate of the mean track should increase each frame
    mean_x = tracks[:, :, 0].mean(axis=1)  # (F,)
    assert np.all(np.diff(mean_x) > 0), f"Expected monotonic rightward drift: {mean_x}"


def test_layer1_no_confident_keypoints_returns_centre():
    """Degenerate case: all keypoints below confidence threshold → centre track."""
    num_frames = 3
    char_size = (64, 64)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    kps[:, :, 2] = 0.1  # all low confidence
    M = np.eye(3, dtype=np.float64)
    tracks = _layer1_skeleton_tracks(kps, M, char_size)
    assert tracks.shape == (num_frames, 1, 2)
    assert np.allclose(tracks, 0.5, atol=0.01)


# ── Layer 2: Gaussian interpolation ──────────────────────────────────────────


def test_layer2_shape():
    num_frames = 5
    density = 64
    n1 = 10
    layer1 = np.random.rand(num_frames, n1, 2).astype(np.float32)
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    result = _layer2_interpolated_tracks(layer1, mask, density)
    assert result.shape[0] == num_frames
    assert result.shape[1] == density // 2
    assert result.shape[2] == 2


def test_layer2_values_in_unit_range():
    num_frames = 4
    n1 = 8
    layer1 = np.random.rand(num_frames, n1, 2).astype(np.float32)
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    result = _layer2_interpolated_tracks(layer1, mask, density=32)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ── Body transform ────────────────────────────────────────────────────────────


def test_body_transform_maps_to_char_space():
    num_frames = 3
    char_size = (128, 128)
    ref_size = (128, 128)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    kps[:, :17, 0] = 40.0
    kps[:, :17, 1] = 50.0
    kps[:, :17, 2] = 0.9
    M = _build_body_transform(kps, ref_size, char_size)
    assert M.shape == (3, 3)
    # Transform a point and check it stays in reasonable range
    pt = np.array([[40.0, 50.0]], dtype=np.float32)
    out = _apply_transform_to_points(pt, M, char_size)
    assert out.shape == (1, 2)
    assert out[0, 0] >= 0.0 and out[0, 0] <= 1.0
    assert out[0, 1] >= 0.0 and out[0, 1] <= 1.0


# ── Nonrigid mask ─────────────────────────────────────────────────────────────


def test_nonrigid_mask_shape():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    nr = _build_nonrigid_mask(mask)
    assert nr.shape == mask.shape
    assert nr.dtype == np.uint8


def test_nonrigid_mask_excludes_body_core():
    """The core body region should be suppressed in the non-rigid mask."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    nr = _build_nonrigid_mask(mask)
    # Centre pixel of body should be 0 (excluded)
    assert nr[64, 64] == 0


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_trajectory_single_frame_raises(tmp_path):
    """Videos with only one frame cannot produce trajectories."""
    vid = _make_video(tmp_path / "motion.mp4", frames=1)
    pose = _make_drifting_pose(num_frames=1)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError, match="at least 2 frames"):
        synthesize_trajectory(pose, seg, vid, cfg)


def test_trajectory_npz_roundtrip(tmp_path):
    """TrajectoryMap produced by synthesize_trajectory must survive save/load."""
    vid = _make_video(tmp_path / "motion.mp4", frames=4)
    pose = _make_drifting_pose(num_frames=4)
    seg = _make_segmentation(tmp_path)
    cfg = _make_config(tmp_path)
    result = synthesize_trajectory(pose, seg, vid, cfg)

    npz_path = tmp_path / "traj.npz"
    result.save(npz_path)
    loaded = TrajectoryMap.load(npz_path)

    assert loaded.density == result.density
    assert loaded.frame_size == result.frame_size
    np.testing.assert_array_almost_equal(loaded.tracks, result.tracks)
