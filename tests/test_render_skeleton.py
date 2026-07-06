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


# Canonical OpenPose 18-joint palette (RGB order) — what VACE saw in training
# via controlnet_aux draw_bodypose. Joint circles are drawn at full palette
# color after the limb layer is dimmed, so exact palette values must appear.
_CANONICAL_PALETTE_RGB = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)


def test_render_skeleton_uses_canonical_openpose_palette(tmp_path):
    pose = _make_pose_sequence(tmp_path, frames=1)
    frames = render_skeleton_frames(pose, size=(128, 128), num_frames=1)
    pixels = {tuple(int(v) for v in px) for px in frames[0].reshape(-1, 3)}

    palette_bgr = {(b, g, r) for (r, g, b) in _CANONICAL_PALETTE_RGB}
    assert len(palette_bgr & pixels) >= 3, (
        "expected >=3 exact canonical OpenPose joint colors in the render"
    )
    # Old renderer drew white joints/limbs — out-of-distribution for VACE.
    assert (255, 255, 255) not in pixels


def test_render_skeleton_draws_synthesized_neck_joint(tmp_path):
    keypoints = np.zeros((1, 133, 3), dtype=np.float32)
    keypoints[0, 5] = (20, 20, 0.95)  # COCO L shoulder
    keypoints[0, 6] = (40, 20, 0.95)  # COCO R shoulder
    pose = PoseSequence(
        source_video_path=tmp_path / "motion.mp4",
        keypoints=keypoints,
        frame_size=(64, 64),
        fps=24.0,
    )
    out_w, out_h = 128, 128
    frames = render_skeleton_frames(pose, size=(out_w, out_h), num_frames=1)

    # Neck = midpoint of the shoulders, scaled into output space.
    nx = round(30 * out_w / 64)
    ny = round(20 * out_h / 64)
    neck_bgr = (0, 85, 255)  # palette[1] (255, 85, 0) RGB -> BGR
    assert tuple(int(v) for v in frames[0][ny, nx]) == neck_bgr


def test_render_skeleton_draws_hand_bones_and_joints(tmp_path):
    # Left-hand keypoints 91..95 along a diagonal (frame 64x64 -> render 128x128,
    # 2x scale). Bones drawn as rainbow lines, joints as red dots (r=4).
    keypoints = np.zeros((1, 133, 3), dtype=np.float32)
    for i, xy in enumerate([(20, 20), (28, 28), (36, 36), (44, 44), (52, 52)]):
        keypoints[0, 91 + i] = (xy[0], xy[1], 0.95)
    pose = PoseSequence(
        source_video_path=tmp_path / "motion.mp4",
        keypoints=keypoints,
        frame_size=(64, 64),
        fps=24.0,
    )
    frames = render_skeleton_frames(pose, size=(128, 128), num_frames=1)
    canvas = frames[0]

    # (48,48) sits on the first bone, >4px from either joint -> pure edge color.
    assert np.any(canvas[48, 48] > 0), "expected a hand bone drawn between joints"
    # Joint 91 scaled to (40,40): red dot BGR(0,0,255).
    assert tuple(int(v) for v in canvas[40, 40]) == (0, 0, 255)


def test_render_skeleton_draws_face_dots(tmp_path):
    pose = _make_pose_sequence(tmp_path, frames=1)
    # Add three face keypoints (COCO-WholeBody 23..90 is the face block).
    pose.keypoints[0, 23] = (30, 25, 0.95)
    pose.keypoints[0, 45] = (32, 27, 0.95)
    pose.keypoints[0, 60] = (34, 29, 0.95)
    frames = render_skeleton_frames(pose, size=(128, 128), num_frames=1)
    pixels = {tuple(int(v) for v in px) for px in frames[0].reshape(-1, 3)}

    assert (255, 255, 255) in pixels, "expected white face dots"


def test_render_skeleton_below_threshold_hands_face_are_noop(tmp_path):
    body_only = _make_pose_sequence(tmp_path, frames=1)
    with_low_conf = _make_pose_sequence(tmp_path, frames=1)
    # Hands + face present but below the 0.3 confidence gate -> must draw nothing.
    for idx in (23, 45, 60, 91, 95, 112, 120):
        with_low_conf.keypoints[0, idx] = (30, 30, 0.1)

    a = render_skeleton_frames(body_only, size=(128, 128), num_frames=1)[0]
    b = render_skeleton_frames(with_low_conf, size=(128, 128), num_frames=1)[0]
    assert np.array_equal(a, b), "sub-threshold hands/face changed the render"


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
