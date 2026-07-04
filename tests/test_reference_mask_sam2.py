"""Mocked SAM-2 reference-video mask propagation tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.extract.reference_mask import (
    _reference_mask_to_vace_frame,
    propagate_reference_masks,
    resample_reference_masks,
    write_vace_reference_mask_video,
)
from motion_mirror.types import PoseSequence, ReferenceMaskResult


def _make_video(path: Path, frames: int = 3, size: tuple[int, int] = (32, 32)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, size)
    for idx in range(frames):
        frame = np.full((size[1], size[0], 3), 30 + idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_pose(frames: int = 3, frame_size: tuple[int, int] = (32, 32)) -> PoseSequence:
    kps = np.zeros((frames, 133, 3), dtype=np.float32)
    for idx in range(frames):
        kps[idx, :17, 0] = np.linspace(10, 20, 17)
        kps[idx, :17, 1] = np.linspace(8, 24, 17)
        kps[idx, :17, 2] = 0.9
    return PoseSequence(
        source_video_path=Path("motion.mp4"),
        keypoints=kps,
        frame_size=frame_size,
        fps=16.0,
    )


class FakeVideoPredictor:
    def __init__(self) -> None:
        self.init_video_path: str | None = None
        self.prompt_kwargs: dict | None = None

    def init_state(self, video_path: str):
        self.init_video_path = video_path
        return {"video_path": video_path}

    def add_new_points_or_box(self, **kwargs):
        self.prompt_kwargs = kwargs
        logits = np.zeros((1, 1, 32, 32), dtype=np.float32)
        logits[:, :, 8:24, 8:24] = 1.0
        return 0, [1], logits

    def propagate_in_video(self, state):
        for frame_idx in range(3):
            logits = np.zeros((1, 1, 32, 32), dtype=np.float32)
            logits[:, :, 8:24, 8 + frame_idx:24 + frame_idx] = 1.0
            yield frame_idx, [1], logits


def test_propagate_reference_masks_uses_mocked_sam2_video_predictor(tmp_path):
    video = _make_video(tmp_path / "motion.mp4")
    pose = _make_pose()
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        backend="mock",
        reference_masker="sam2",
        device="cpu",
    )
    fake_predictor = FakeVideoPredictor()

    with patch(
        "motion_mirror.extract.reference_mask._get_sam2_video_predictor",
        return_value=fake_predictor,
    ):
        result = propagate_reference_masks(video, pose, cfg)

    assert isinstance(result, ReferenceMaskResult)
    assert result.masks.shape == (3, 32, 32)
    assert result.masks.dtype == np.uint8
    assert result.masks[0, 12, 12] == 255
    assert result.mask_video_path is not None
    assert result.mask_video_path.exists()
    assert fake_predictor.init_video_path is not None
    assert fake_predictor.prompt_kwargs is not None
    assert "box" in fake_predictor.prompt_kwargs


def test_resample_reference_masks_resizes_and_resamples():
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 255
    masks[1, 2:4, 2:4] = 255
    ref = ReferenceMaskResult(
        source_video_path=Path("motion.mp4"),
        mask_video_path=None,
        masks=masks,
        frame_size=(4, 4),
        fps=16.0,
    )

    out = resample_reference_masks(ref, num_frames=4, size=(8, 8))

    assert out.shape == (4, 8, 8)
    assert out.dtype == np.uint8
    assert out[0, 2:6, 2:6].max() == 255


def test_reference_mask_to_vace_frame_keeps_subject_white():
    """VACE mask polarity: white = generate. The subject must stay white so
    generation happens where the person is; a black subject makes VACE copy
    the control video's skeleton pixels through verbatim (2026-07-04 GPU run).
    """
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    frame = _reference_mask_to_vace_frame(mask)

    assert frame[1, 1, 0] == 255
    assert frame[0, 0, 0] == 0


def test_write_vace_reference_mask_video_uses_vace_polarity(tmp_path):
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 255
    ref = ReferenceMaskResult(
        source_video_path=Path("motion.mp4"),
        mask_video_path=None,
        masks=masks,
        frame_size=(4, 4),
        fps=16.0,
    )
    captured: list[np.ndarray] = []

    def fake_write(path, frames, fps):
        captured.extend(frames)

    with patch("motion_mirror.extract.reference_mask._write_bgr_video", side_effect=fake_write):
        write_vace_reference_mask_video(
            ref,
            tmp_path / "conditioning_mask.mp4",
            size=(4, 4),
            num_frames=1,
        )

    assert captured
    assert captured[0][1, 1, 0] == 255
    assert captured[0][0, 0, 0] == 0
