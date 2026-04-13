import tempfile
from pathlib import Path

import numpy as np
import pytest

from motion_mirror.types import (
    GenerationResult,
    PoseSequence,
    SegmentationResult,
    TrajectoryMap,
)


def test_segmentation_result_shapes():
    seg = SegmentationResult(
        source_image_path=Path("img.png"),
        rgba_path=Path("img_rgba.png"),
        mask=np.zeros((64, 64), dtype=np.uint8),
        rgba=np.zeros((64, 64, 4), dtype=np.uint8),
    )
    assert seg.mask.shape == (64, 64)
    assert seg.mask.dtype == np.uint8
    assert seg.rgba.shape == (64, 64, 4)
    assert seg.rgba.dtype == np.uint8


def test_pose_sequence_shapes():
    pose = PoseSequence(
        source_video_path=Path("vid.mp4"),
        keypoints=np.zeros((10, 133, 3), dtype=np.float32),
        frame_size=(640, 480),
        fps=24.0,
    )
    assert pose.keypoints.shape == (10, 133, 3)
    assert pose.keypoints.dtype == np.float32
    assert pose.frame_size == (640, 480)
    assert pose.fps == 24.0


def test_trajectory_map_shapes():
    traj = TrajectoryMap(
        tracks=np.zeros((5, 512, 2), dtype=np.float32),
        flow_fields=np.zeros((4, 64, 64, 2), dtype=np.float32),
        density=512,
        frame_size=(832, 480),
    )
    assert traj.tracks.shape == (5, 512, 2)
    assert traj.flow_fields.shape == (4, 64, 64, 2)
    assert traj.density == 512
    assert traj.frame_size == (832, 480)


def test_trajectory_map_save_load_roundtrip():
    rng = np.random.default_rng(0)
    tracks = rng.random((5, 512, 2), dtype=np.float32)  # type: ignore[call-overload]
    flows = rng.random((4, 32, 32, 2), dtype=np.float32)  # type: ignore[call-overload]

    traj = TrajectoryMap(
        tracks=tracks.astype(np.float32),
        flow_fields=flows.astype(np.float32),
        density=512,
        frame_size=(832, 480),
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "traj.npz"
        traj.save(p)
        assert p.exists()

        loaded = TrajectoryMap.load(p)
        assert loaded.density == 512
        assert loaded.frame_size == (832, 480)
        assert loaded.tracks.shape == (5, 512, 2)
        assert loaded.flow_fields.shape == (4, 32, 32, 2)
        np.testing.assert_array_almost_equal(loaded.tracks, traj.tracks)
        np.testing.assert_array_almost_equal(loaded.flow_fields, traj.flow_fields)


def test_trajectory_map_values_in_range():
    rng = np.random.default_rng(1)
    tracks = rng.random((3, 128, 2)).astype(np.float32)
    traj = TrajectoryMap(
        tracks=tracks,
        flow_fields=np.zeros((2, 16, 16, 2), dtype=np.float32),
        density=128,
        frame_size=(256, 256),
    )
    assert traj.tracks.min() >= 0.0
    assert traj.tracks.max() <= 1.0


def test_generation_result():
    gen = GenerationResult(
        video_path=Path("out.mp4"),
        backend="mock",
        resolution="832x480",
        num_frames=81,
    )
    assert gen.num_frames == 81
    assert gen.backend == "mock"
    assert gen.video_path == Path("out.mp4")
