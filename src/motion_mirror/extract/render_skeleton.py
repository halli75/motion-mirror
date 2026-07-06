"""Render pose keypoints into conditioning videos for Wan VACE."""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np

from ..types import PoseSequence

# VACE was trained on canonical OpenPose whole-body renders (controlnet_aux
# draw_bodypose + draw_handpose + draw_facepose): 18 body joints including a
# synthesized neck, a fixed limb sequence and 18-color palette (limbs drawn as
# filled ellipses then dimmed to 0.6 before full-color joint circles); each
# hand as 20 rainbow-colored bones + red joint dots; the face as white dots.
# Anything else (a body-only render, or the older COCO-17 custom-color render)
# is out-of-distribution — the model then reproduces the control literally, or
# hallucinates the missing hands/face instead of following them.

# OpenPose BODY-18 index <- COCO-17 index; -1 marks the synthesized neck
# (midpoint of the two shoulders, confidence = min of both).
_OPENPOSE_FROM_COCO: tuple[int, ...] = (
    0,   # 0  nose
    -1,  # 1  neck (synthesized)
    6,   # 2  R shoulder
    8,   # 3  R elbow
    10,  # 4  R wrist
    5,   # 5  L shoulder
    7,   # 6  L elbow
    9,   # 7  L wrist
    12,  # 8  R hip
    14,  # 9  R knee
    16,  # 10 R ankle
    11,  # 11 L hip
    13,  # 12 L knee
    15,  # 13 L ankle
    2,   # 14 R eye
    1,   # 15 L eye
    4,   # 16 R ear
    3,   # 17 L ear
)

# Canonical limb sequence (0-indexed BODY-18 pairs), same order as the palette.
_LIMB_SEQ: tuple[tuple[int, int], ...] = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)

# Canonical OpenPose palette, stored BGR for cv2 (source values are RGB).
_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = tuple(
    (b, g, r) for (r, g, b) in _PALETTE_RGB
)

# COCO-WholeBody 133-keypoint layout: 0-16 body (mapped above), 17-22 feet
# (unused — BODY-18 has no feet), 23-90 face (68 pts), 91-111 left hand,
# 112-132 right hand (21 pts each).
_FACE_SLICE = slice(23, 91)
_LEFT_HAND_SLICE = slice(91, 112)
_RIGHT_HAND_SLICE = slice(112, 133)

# 21-point hand skeleton, 20 bones (controlnet_aux draw_handpose edge order).
_HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# Per-bone HSV rainbow, hsv_to_rgb(i/20, 1, 1) — stdlib colorsys matches
# matplotlib.colors.hsv_to_rgb used by controlnet_aux. Stored BGR for cv2 so
# the on-canvas color matches controlnet's RGB-canvas render visually.
_HAND_EDGE_COLOURS_BGR: tuple[tuple[int, int, int], ...] = tuple(
    (round(b * 255), round(g * 255), round(r * 255))
    for (r, g, b) in (colorsys.hsv_to_rgb(i / 20.0, 1.0, 1.0) for i in range(20))
)

# Hand joint dots and face dots (BGR).
_HAND_JOINT_BGR = (0, 0, 255)  # red
_FACE_DOT_BGR = (255, 255, 255)  # white


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

    stick_width = max(2, round(min(out_w, out_h) / 160))
    joint_radius = max(2, round(min(out_w, out_h) / 120))
    frames: list[np.ndarray] = []

    for frame_kps in keypoints:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        body = _coco_to_openpose18(frame_kps)
        drawn = False

        # Limbs: filled ellipses at full palette color, then dim the whole
        # limb layer to 0.6 — matches controlnet_aux draw_bodypose.
        for colour, (idx0, idx1) in zip(_PALETTE_BGR, _LIMB_SEQ):
            kp0 = body[idx0]
            kp1 = body[idx1]
            if kp0[2] < confidence_threshold or kp1[2] < confidence_threshold:
                continue
            x0, y0 = _scale_point(kp0[:2], src_w, src_h, out_w, out_h)
            x1, y1 = _scale_point(kp1[:2], src_w, src_h, out_w, out_h)
            centre = (round((x0 + x1) / 2), round((y0 + y1) / 2))
            length = float(np.hypot(x1 - x0, y1 - y0))
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            polygon = cv2.ellipse2Poly(
                centre, (max(1, round(length / 2)), stick_width), round(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colour)
            drawn = True

        canvas = (canvas.astype(np.float32) * 0.6).astype(np.uint8)

        # Joints: full-color circles on top of the dimmed limbs.
        for colour, kp in zip(_PALETTE_BGR, body):
            if kp[2] < confidence_threshold:
                continue
            pt = _scale_point(kp[:2], src_w, src_h, out_w, out_h)
            cv2.circle(canvas, pt, joint_radius, colour, -1)
            drawn = True

        # Hands + face at full brightness, after the dim (controlnet_aux
        # draws hands/face on the already-dimmed body layer). Zero-conf
        # keypoints (mock / undetected) are skipped by the threshold, so
        # body-only sequences render exactly as before.
        _draw_hand(canvas, frame_kps[_LEFT_HAND_SLICE], (src_w, src_h), (out_w, out_h), confidence_threshold)
        _draw_hand(canvas, frame_kps[_RIGHT_HAND_SLICE], (src_w, src_h), (out_w, out_h), confidence_threshold)
        _draw_face(canvas, frame_kps[_FACE_SLICE], (src_w, src_h), (out_w, out_h), confidence_threshold)

        if not drawn:
            cv2.circle(
                canvas,
                (out_w // 2, out_h // 2),
                max(joint_radius, 3),
                _PALETTE_BGR[0],
                -1,
            )

        frames.append(canvas)

    return frames


def _coco_to_openpose18(frame_kps: np.ndarray) -> np.ndarray:
    """Remap COCO-WholeBody's first 17 body keypoints to OpenPose BODY-18."""
    body = np.zeros((18, 3), dtype=np.float32)
    for op_idx, coco_idx in enumerate(_OPENPOSE_FROM_COCO):
        if coco_idx >= 0:
            body[op_idx] = frame_kps[coco_idx]
    l_sho = frame_kps[5]
    r_sho = frame_kps[6]
    body[1, :2] = (l_sho[:2] + r_sho[:2]) / 2.0
    body[1, 2] = min(float(l_sho[2]), float(r_sho[2]))
    return body


def _draw_hand(
    canvas: np.ndarray,
    hand_kps: np.ndarray,
    src_wh: tuple[int, int],
    out_wh: tuple[int, int],
    threshold: float,
) -> None:
    """Draw one 21-point hand: rainbow bones (controlnet_aux draw_handpose)."""
    src_w, src_h = src_wh
    out_w, out_h = out_wh
    for colour, (a, b) in zip(_HAND_EDGE_COLOURS_BGR, _HAND_EDGES):
        if hand_kps[a][2] < threshold or hand_kps[b][2] < threshold:
            continue
        p0 = _scale_point(hand_kps[a][:2], src_w, src_h, out_w, out_h)
        p1 = _scale_point(hand_kps[b][:2], src_w, src_h, out_w, out_h)
        cv2.line(canvas, p0, p1, colour, thickness=2)
    for kp in hand_kps:
        if kp[2] < threshold:
            continue
        pt = _scale_point(kp[:2], src_w, src_h, out_w, out_h)
        cv2.circle(canvas, pt, 4, _HAND_JOINT_BGR, thickness=-1)


def _draw_face(
    canvas: np.ndarray,
    face_kps: np.ndarray,
    src_wh: tuple[int, int],
    out_wh: tuple[int, int],
    threshold: float,
) -> None:
    """Draw the 68-point face as white dots (controlnet_aux draw_facepose)."""
    src_w, src_h = src_wh
    out_w, out_h = out_wh
    for kp in face_kps:
        if kp[2] < threshold:
            continue
        pt = _scale_point(kp[:2], src_w, src_h, out_w, out_h)
        cv2.circle(canvas, pt, 3, _FACE_DOT_BGR, thickness=-1)


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
    # WanVACE mask convention: white (255) = generate this region, black (0) =
    # preserve the control `video` verbatim. The skeleton is passed as the
    # control video, so the whole frame must be marked for generation —
    # otherwise the black skeleton lines get copied straight into the output
    # (the "skeleton-as-output" failure). A full-white mask lets the model use
    # the skeleton purely as a structural hint while synthesizing the character
    # supplied via reference_images.
    return np.full_like(frame, np.uint8(255))


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
