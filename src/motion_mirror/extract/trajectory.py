"""Trajectory synthesis — the core algorithmic stage of Motion Mirror.

Three-layer approach (all CPU / pure numpy+opencv, no GPU required):

  Layer 1 — Skeleton keypoint trajectories
    Direct tracks from confident DWPose keypoints, coordinate-normalised
    into character-image space via a similarity transform.

  Layer 2 — Gaussian-falloff interpolation
    Additional tracks seeded uniformly inside the subject mask, each
    moved by a weighted sum of Layer-1 displacements where the weights
    fall off as a Gaussian in normalised space (σ=0.15).

  Layer 3 — Optical-flow tracks for non-rigid regions
    cv2.calcOpticalFlowFarneback computed frame-0 → frame-k directly
    (never chained) for seed points in hair/clothing regions.

Camera motion is compensated first by computing a per-frame homography
on the background pixels and warping frames into frame-0 space before
the optical-flow step.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import PoseSequence, SegmentationResult, TrajectoryMap

# ── Constants ─────────────────────────────────────────────────────────────────

# COCO-WholeBody body keypoint indices (0-16 = 17 COCO body joints)
_BODY_KP_INDICES = list(range(17))

# Gaussian falloff sigma in normalised [0,1] space
_GAUSSIAN_SIGMA = 0.15

# Farneback optical flow parameters (frame-0 → frame-k)
_FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Dilation kernel sizes
_MASK_DILATE_BG_PX = 20   # isolate background for homography
_MASK_DILATE_NR_PX = 30   # non-rigid region border around subject


# ── Public API ────────────────────────────────────────────────────────────────


def synthesize_trajectory(
    pose_seq: PoseSequence,
    segmentation: SegmentationResult,
    motion_video_path: Path,
    config: MotionMirrorConfig | None = None,
) -> TrajectoryMap:
    """Build a dense trajectory map for use with the generation backend.

    Parameters
    ----------
    pose_seq:
        Output of ``extract_pose()``.
    segmentation:
        Output of ``segment_subject()`` on the *character image*.
    motion_video_path:
        Path to the reference motion video (same video used for pose).
    config:
        Pipeline configuration.  Defaults created if omitted.

    Returns
    -------
    TrajectoryMap
        ``tracks`` shape: ``(F, density, 2)`` float32, values in ``[0, 1]``
        character-image space.
        ``flow_fields`` shape: ``(F-1, H, W, 2)`` float32.
    """
    cfg = config or MotionMirrorConfig()
    density = cfg.trajectory_density

    char_w, char_h = segmentation.mask.shape[1], segmentation.mask.shape[0]
    char_size = (char_w, char_h)

    # --- Load video frames --------------------------------------------------
    frames = _load_frames(motion_video_path)
    num_frames = len(frames)

    if num_frames < 2:
        raise ValueError(
            f"Video must have at least 2 frames for trajectory synthesis, "
            f"got {num_frames} from {motion_video_path}"
        )

    # Ensure pose sequence matches frame count (truncate to shorter)
    kps = pose_seq.keypoints  # (F, 133, 3)
    num_frames = min(num_frames, kps.shape[0])
    frames = frames[:num_frames]
    kps = kps[:num_frames]

    # --- Camera motion compensation ----------------------------------------
    stabilized_frames, _ = _compensate_camera_motion(frames, segmentation.mask)

    # --- Coordinate transform: reference video → character image space ------
    body_transform = _build_body_transform(
        kps, pose_seq.frame_size, char_size
    )

    # --- Layer 1: skeleton keypoint tracks ----------------------------------
    layer1 = _layer1_skeleton_tracks(kps, body_transform, char_size)
    # layer1: (F, N1, 2) normalised [0,1]

    # --- Layer 2: Gaussian-falloff interpolated tracks ----------------------
    layer2 = _layer2_interpolated_tracks(layer1, segmentation.mask, density)
    # layer2: (F, N2, 2)

    # --- Layer 3: optical-flow tracks for non-rigid regions -----------------
    layer3, flow_fields = _layer3_flow_tracks(
        stabilized_frames,
        segmentation.mask,
        body_transform,
        char_size,
        density,
    )
    # layer3: (F, N3, 2), flow_fields: (F-1, H, W, 2)

    # --- Compose layers and subsample to exactly `density` tracks -----------
    all_tracks = np.concatenate([layer1, layer2, layer3], axis=1)  # (F, N, 2)

    if all_tracks.shape[1] > density:
        rng = np.random.default_rng(0)
        idx = rng.choice(all_tracks.shape[1], density, replace=False)
        all_tracks = all_tracks[:, idx, :]
    elif all_tracks.shape[1] < density:
        # Pad by repeating existing tracks with small jitter
        shortage = density - all_tracks.shape[1]
        rng = np.random.default_rng(1)
        pad_idx = rng.choice(all_tracks.shape[1], shortage, replace=True)
        jitter = rng.normal(0, 0.001, (num_frames, shortage, 2)).astype(np.float32)
        pad = all_tracks[:, pad_idx, :] + jitter
        pad = np.clip(pad, 0.0, 1.0)
        all_tracks = np.concatenate([all_tracks, pad], axis=1)

    all_tracks = np.clip(all_tracks, 0.0, 1.0).astype(np.float32)

    return TrajectoryMap(
        tracks=all_tracks,
        flow_fields=flow_fields,
        density=density,
        frame_size=char_size,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from *video_path* into a list of BGR uint8 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, kernel)


def _compensate_camera_motion(
    frames: list[np.ndarray],
    subject_mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Warp each frame into frame-0 coordinate space using background homography.

    Returns (stabilized_frames, homographies) where homographies[0] is None
    (frame-0 is the reference) and homographies[k] is the 3×3 matrix H_k.
    """
    ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    # Background mask: invert subject, dilate to avoid subject edge contamination
    dilated = _dilate_mask(subject_mask, _MASK_DILATE_BG_PX)
    bg_mask = np.where(dilated == 0, np.uint8(255), np.uint8(0))

    orb = cv2.ORB_create(nfeatures=500)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, bg_mask)

    stabilized: list[np.ndarray] = [frames[0]]
    homographies: list[np.ndarray | None] = [None]

    h, w = frames[0].shape[:2]

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, bg_mask)

        H: np.ndarray | None = None
        warped = frame

        if des_ref is not None and des is not None and len(kp_ref) >= 4 and len(kp) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_ref, des)
            matches = sorted(matches, key=lambda m: m.distance)

            if len(matches) >= 4:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    warped = cv2.warpPerspective(frame, H, (w, h))

        stabilized.append(warped)
        homographies.append(H)

    return stabilized, homographies


def _build_body_transform(
    keypoints: np.ndarray,
    ref_frame_size: tuple[int, int],
    char_image_size: tuple[int, int],
) -> np.ndarray:
    """Build a 3×3 similarity transform (scale + translate) mapping reference
    body bounding box → character-image body bounding box.

    If no confident keypoints are found, returns an identity-like transform
    scaled to fit the frame.
    """
    ref_w, ref_h = ref_frame_size
    char_w, char_h = char_image_size

    # Gather confident body keypoints across all frames
    body_kps = keypoints[:, _BODY_KP_INDICES, :]  # (F, 17, 3)
    conf = body_kps[:, :, 2]
    mask = conf > 0.3  # (F, 17)

    if mask.sum() < 2:
        # Fallback: scale entire frame to char image size
        sx = char_w / ref_w
        sy = char_h / ref_h
        M = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        return M

    xs = body_kps[:, :, 0][mask]
    ys = body_kps[:, :, 1][mask]

    ref_x1, ref_x2 = xs.min(), xs.max()
    ref_y1, ref_y2 = ys.min(), ys.max()

    ref_bw = max(ref_x2 - ref_x1, 1.0)
    ref_bh = max(ref_y2 - ref_y1, 1.0)

    # Character image body bbox: use 80% of image centred
    char_margin_x = char_w * 0.1
    char_margin_y = char_h * 0.1
    char_x1 = char_margin_x
    char_y1 = char_margin_y
    char_bw = char_w - 2 * char_margin_x
    char_bh = char_h - 2 * char_margin_y

    sx = char_bw / ref_bw
    sy = char_bh / ref_bh
    tx = char_x1 - ref_x1 * sx
    ty = char_y1 - ref_y1 * sy

    M = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)
    return M


def _apply_transform_to_points(
    pts: np.ndarray,  # (N, 2) float32
    M: np.ndarray,    # 3x3
    char_size: tuple[int, int],
) -> np.ndarray:
    """Apply homography M to points and normalise to [0,1]."""
    char_w, char_h = char_size
    if len(pts) == 0:
        return pts
    pts_h = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts_h, M.astype(np.float32))
    out = out.reshape(-1, 2)
    out[:, 0] /= char_w
    out[:, 1] /= char_h
    return out.astype(np.float32)


def _layer1_skeleton_tracks(
    keypoints: np.ndarray,        # (F, 133, 3)
    body_transform: np.ndarray,   # 3x3
    char_size: tuple[int, int],
) -> np.ndarray:
    """Return (F, N1, 2) normalised tracks for confident keypoints."""
    num_frames = keypoints.shape[0]

    # Select keypoints with mean confidence > 0.3 across frames
    mean_conf = keypoints[:, :, 2].mean(axis=0)  # (133,)
    confident = np.where(mean_conf > 0.3)[0]

    if len(confident) == 0:
        # Degenerate fallback: single track at image centre
        return np.full((num_frames, 1, 2), 0.5, dtype=np.float32)

    tracks = np.zeros((num_frames, len(confident), 2), dtype=np.float32)
    for f in range(num_frames):
        pts = keypoints[f, confident, :2]  # (N1, 2)
        tracks[f] = _apply_transform_to_points(pts, body_transform, char_size)

    return np.clip(tracks, 0.0, 1.0)


def _layer2_interpolated_tracks(
    layer1_tracks: np.ndarray,   # (F, N1, 2) normalised
    mask: np.ndarray,             # (H, W) uint8
    density: int,
) -> np.ndarray:
    """Return (F, N2, 2) normalised tracks via Gaussian-falloff interpolation."""
    num_frames, n1, _ = layer1_tracks.shape
    n2 = density // 2

    # Sample seed points within subject mask
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return np.full((num_frames, n2, 2), 0.5, dtype=np.float32)

    h, w = mask.shape
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ys), size=min(n2, len(ys)), replace=len(ys) < n2)
    seed_x = xs[idx].astype(np.float32) / w  # normalised [0,1]
    seed_y = ys[idx].astype(np.float32) / h
    seeds = np.stack([seed_x, seed_y], axis=1)  # (n2, 2)

    actual_n2 = seeds.shape[0]

    # Displacements from frame-0 anchor
    anchor = layer1_tracks[0]  # (N1, 2)

    tracks = np.zeros((num_frames, actual_n2, 2), dtype=np.float32)
    tracks[0] = seeds

    for f in range(1, num_frames):
        delta = layer1_tracks[f] - anchor  # (N1, 2)
        for i in range(actual_n2):
            p = seeds[i]  # (2,)
            # Gaussian weight from seed to each Layer-1 anchor
            diff = anchor - p[np.newaxis, :]  # (N1, 2)
            dist2 = (diff ** 2).sum(axis=1)   # (N1,)
            weights = np.exp(-dist2 / (2 * _GAUSSIAN_SIGMA ** 2))
            w_sum = weights.sum()
            if w_sum < 1e-12:
                tracks[f, i] = p
            else:
                weights /= w_sum
                displacement = (weights[:, np.newaxis] * delta).sum(axis=0)
                tracks[f, i] = np.clip(p + displacement, 0.0, 1.0)

    return tracks.astype(np.float32)


def _layer3_flow_tracks(
    stabilized_frames: list[np.ndarray],  # list of (H, W, 3) BGR
    subject_mask: np.ndarray,             # (H, W) uint8
    body_transform: np.ndarray,           # 3x3
    char_size: tuple[int, int],
    density: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((F, N3, 2), (F-1, H, W, 2)) optical-flow tracks for non-rigid regions.

    Flow is computed frame-0 → frame-k directly (never chained).
    """
    num_frames = len(stabilized_frames)
    char_w, char_h = char_size
    n3 = density // 4

    ref_gray = cv2.cvtColor(stabilized_frames[0], cv2.COLOR_BGR2GRAY)
    fh, fw = ref_gray.shape

    # Non-rigid region: dilated mask minus dilated body bbox (hair, clothing)
    nr_mask = _build_nonrigid_mask(subject_mask)

    # Sample seed points in the non-rigid region (in video frame space)
    nr_ys, nr_xs = np.where(nr_mask > 127)

    if len(nr_ys) > 0:
        rng = np.random.default_rng(7)
        sel = rng.choice(len(nr_ys), size=min(n3, len(nr_ys)), replace=len(nr_ys) < n3)
        seed_x = nr_xs[sel].astype(np.float32)
        seed_y = nr_ys[sel].astype(np.float32)
    else:
        # Fallback: uniform grid
        seed_x = np.linspace(0, fw - 1, n3, dtype=np.float32)
        seed_y = np.linspace(0, fh - 1, n3, dtype=np.float32)

    actual_n3 = len(seed_x)
    seeds_vid = np.stack([seed_x, seed_y], axis=1)  # (n3, 2) in video px

    # Convert seeds to character-image normalised space
    seeds_char = _apply_transform_to_points(seeds_vid, body_transform, char_size)

    tracks = np.zeros((num_frames, actual_n3, 2), dtype=np.float32)
    tracks[0] = seeds_char

    flow_fields: list[np.ndarray] = []

    for k in range(1, num_frames):
        cur_gray = cv2.cvtColor(stabilized_frames[k], cv2.COLOR_BGR2GRAY)
        # Frame-0 → frame-k directly (spec requirement)
        flow = cv2.calcOpticalFlowFarneback(ref_gray, cur_gray, None, **_FB_PARAMS)
        flow_fields.append(flow.astype(np.float32))  # (H, W, 2)

        # Sample flow at seed positions (clip to valid range)
        xi = np.clip(seed_x.astype(np.int32), 0, fw - 1)
        yi = np.clip(seed_y.astype(np.int32), 0, fh - 1)
        flow_at_seeds = flow[yi, xi, :]  # (n3, 2) in video px

        # Map displaced video-space points to character space
        moved_vid = seeds_vid + flow_at_seeds
        moved_char = _apply_transform_to_points(moved_vid, body_transform, char_size)
        tracks[k] = np.clip(moved_char, 0.0, 1.0)

    if not flow_fields:
        # Single-frame edge case: return zero flow
        flow_fields = [np.zeros((fh, fw, 2), dtype=np.float32)]

    return tracks.astype(np.float32), np.stack(flow_fields).astype(np.float32)


def _build_nonrigid_mask(subject_mask: np.ndarray) -> np.ndarray:
    """Build a mask for non-rigid regions (hair, clothing edges).

    Strategy: dilate subject mask → subtract body bounding-box interior.
    This isolates the soft boundary regions.
    """
    dilated = _dilate_mask(subject_mask, _MASK_DILATE_NR_PX)

    # Body bbox from the coarse subject mask
    ys, xs = np.where(subject_mask > 127)
    nr = dilated.copy()

    if len(ys) > 0:
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        # Erode inward by 40% to get the "core body" region to exclude
        margin_y = max(int((y2 - y1) * 0.2), 1)
        margin_x = max(int((x2 - x1) * 0.2), 1)
        nr[y1 + margin_y: y2 - margin_y, x1 + margin_x: x2 - margin_x] = 0

    return nr
