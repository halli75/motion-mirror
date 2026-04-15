from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from motion_mirror.extract.render_skeleton import (
    render_skeleton_conditioning_artifacts,
    render_skeleton_frames,
)
from motion_mirror.types import PoseSequence


def _make_pose_sequence(tmp_path: Path, frames: int = 3) -> PoseSequence:
    keypoints = np.zeros((frames, 133, 3), dtype=np.float32)
    for idx in range(frames):
        keypoints[idx, 5] = (20 + idx * 2, 20, 0.95)
        keypoints[idx, 6] = (40 + idx * 2, 20, 0.95)
        keypoints[idx, 11] = (22 + idx * 2, 45, 0.95)
        keypoints[idx, 12] = (38 + idx * 2, 45, 0.95)
        keypoints[idx, 13] = (24 + idx * 2, 58, 0.95)
        keypoints[idx, 14] = (36 + idx * 2, 58, 0.95)
    return PoseSequence(
        source_video_path=tmp_path / "motion.mp4",
        keypoints=keypoints,
        frame_size=(64, 64),
        fps=24.0,
    )


def test_render_skeleton_frames_resamples_and_draws(tmp_path):
    pose = _make_pose_sequence(tmp_path, frames=3)
    frames = render_skeleton_frames(pose, size=(80, 60), num_frames=5)

    assert len(frames) == 5
    assert all(frame.shape == (60, 80, 3) for frame in frames)
    assert any(np.any(frame > 0) for frame in frames)


def test_render_skeleton_conditioning_artifacts_writes_readable_videos(tmp_path):
    pose = _make_pose_sequence(tmp_path, frames=4)
    video_path = tmp_path / "pose.mp4"
    mask_path = tmp_path / "mask.mp4"

    render_skeleton_conditioning_artifacts(
        pose_seq=pose,
        video_path=video_path,
        mask_path=mask_path,
        size=(64, 64),
        num_frames=4,
    )

    assert video_path.exists()
    assert mask_path.exists()

    cap_video = cv2.VideoCapture(str(video_path))
    cap_mask = cv2.VideoCapture(str(mask_path))
    assert cap_video.isOpened()
    assert cap_mask.isOpened()
    assert int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap_video.release()
    cap_mask.release()
