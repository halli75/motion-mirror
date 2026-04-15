"""Render pose keypoints into conditioning videos for Wan VACE."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import PoseSequence

_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

_EDGE_COLOURS: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),
    (255, 255, 255),
    (255, 200, 0),
    (255, 200, 0),
    (0, 220, 255),
    (0, 220, 255),
    (180, 180, 180),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 0, 255),
    (180, 180, 180),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
)


def render_skeleton_frames(
    pose_seq: PoseSequence,
    size: tuple[int, int],
    num_frames: int | None = None,
    confidence_threshold: float = 0.3,
) -> list[np.ndarray]:
    """Render pose keypoints into BGR skeleton frames."""
    out_w, out_h = size
    src_w, src_h = pose_seq.frame_size
    keypoints = _resample_keypoints(pose_seq.keypoints, num_frames)

    line_thickness = max(2, round(min(out_w, out_h) / 160))
    joint_radius = max(2, round(min(out_w, out_h) / 120))
    frames: list[np.ndarray] = []

    for frame_kps in keypoints:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        drawn = False

        for colour, (idx0, idx1) in zip(_EDGE_COLOURS, _BODY_EDGES):
            kp0 = frame_kps[idx0]
            kp1 = frame_kps[idx1]
            if kp0[2] < confidence_threshold or kp1[2] < confidence_threshold:
                continue
            pt0 = _scale_point(kp0[:2], src_w, src_h, out_w, out_h)
            pt1 = _scale_point(kp1[:2], src_w, src_h, out_w, out_h)
            cv2.line(canvas, pt0, pt1, colour, line_thickness, lineType=cv2.LINE_AA)
            drawn = True

        for kp in frame_kps[:17]:
            if kp[2] < confidence_threshold:
                continue
            pt = _scale_point(kp[:2], src_w, src_h, out_w, out_h)
            cv2.circle(canvas, pt, joint_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            drawn = True

        if not drawn:
            cv2.circle(
                canvas,
                (out_w // 2, out_h // 2),
                max(joint_radius, 3),
                (255, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

        frames.append(canvas)

    return frames


def render_skeleton_conditioning_artifacts(
    pose_seq: PoseSequence,
    video_path: Path,
    mask_path: Path,
    size: tuple[int, int],
    num_frames: int,
    fps: float = 16.0,
) -> tuple[Path, Path]:
    """Write skeleton conditioning video and matching VACE mask video."""
    frames = render_skeleton_frames(pose_seq, size=size, num_frames=num_frames)
    mask_frames = [_build_mask_frame(frame) for frame in frames]

    _write_video(video_path, frames, fps=fps)
    _write_video(mask_path, mask_frames, fps=fps)
    return video_path, mask_path


def _resample_keypoints(keypoints: np.ndarray, num_frames: int | None) -> np.ndarray:
    if num_frames is None or keypoints.shape[0] == num_frames:
        return keypoints
    if keypoints.shape[0] == 0:
        raise ValueError("PoseSequence contains no frames")

    indices = np.linspace(0, keypoints.shape[0] - 1, num_frames).round().astype(np.int32)
    return keypoints[indices]


def _scale_point(
    point_xy: np.ndarray,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
) -> tuple[int, int]:
    x = int(np.clip(round(float(point_xy[0]) * out_w / max(src_w, 1)), 0, out_w - 1))
    y = int(np.clip(round(float(point_xy[1]) * out_h / max(src_h, 1)), 0, out_h - 1))
    return x, y


def _build_mask_frame(frame: np.ndarray) -> np.ndarray:
    skeleton_pixels = np.any(frame > 0, axis=2)
    mask = np.where(skeleton_pixels, np.uint8(0), np.uint8(255))
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("No frames provided for skeleton conditioning video")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {path}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Conditioning frames must all have the same size")
            writer.write(frame)
    finally:
        writer.release()
