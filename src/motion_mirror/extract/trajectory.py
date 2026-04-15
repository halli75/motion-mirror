"""Trajectory synthesis — the core algorithmic stage of Motion Mirror.

Three-layer approach (all CPU / pure numpy+opencv, no GPU required for
the default Farneback estimator):

  Layer 1 — Skeleton keypoint trajectories
    Direct tracks from confident DWPose keypoints, coordinate-normalised
    into character-image space via a similarity transform.

  Layer 2 — Gaussian-falloff interpolation
    Additional tracks seeded uniformly inside the subject mask, each
    moved by a weighted sum of Layer-1 displacements where the weights
    fall off as a Gaussian in normalised space (σ=0.15).

  Layer 3 — Optical-flow tracks for non-rigid regions
    Flow computed frame-0 → frame-k directly (never chained) for seed
    points in hair/clothing regions.  Two estimators are supported:

      farneback (default) — cv2.calcOpticalFlowFarneback, pure CPU.
      raft               — torchvision RAFT-Large (v0.2a, GPU recommended).
                           Weights auto-download via torchvision (~20 MB).
                           Falls back to Farneback if torchvision is absent
                           or no GPU is available.

Camera motion is compensated first by computing a per-frame homography
on the background pixels and warping frames into frame-0 space before
the optical-flow step.
"""
from __future__ import annotations

import warnings
from pathlib import Path

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

# Module-level RAFT model cache — keyed by device string so we don't reload
# when the same device is requested repeatedly.
_raft_cache: dict[str, object] = {}

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

    char_size = (segmentation.mask.shape[1], segmentation.mask.shape[0])  # (W, H)

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

    # --- Temporal resampling to cfg.num_frames --------------------------------
    # Map however many source frames we have onto the target output length.
    target_frames = cfg.num_frames
    if target_frames > 0 and num_frames != target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames).round().astype(np.int32)
        frames = [frames[i] for i in indices]
        kps = kps[indices]
        num_frames = target_frames

    # --- Build video-space body mask from frame-0 pose keypoints -------------
    # Camera stabilization and Layer-3 seed selection must operate in the
    # reference video's coordinate space, not the character image space.
    # We derive the mask from pose keypoints so it is always correctly sized.
    fh, fw = frames[0].shape[:2]
    video_body_mask = _build_video_body_mask(kps[0], (fh, fw))

    # --- Camera motion compensation ----------------------------------------
    stabilized_frames, homographies = _compensate_camera_motion(frames, video_body_mask)

    # Warn if very few frames were successfully stabilized
    valid_h_count = sum(1 for h in homographies[1:] if h is not None)
    total_pairs = num_frames - 1
    if total_pairs > 1 and valid_h_count < total_pairs // 2:
        warnings.warn(
            f"Camera stabilization: only {valid_h_count}/{total_pairs} frames "
            "stabilized (insufficient background features). "
            "Trajectory quality may be degraded for non-static cameras.",
            UserWarning,
            stacklevel=2,
        )

    # --- Coordinate transform: reference video → character image space ------
    body_transform = _build_body_transform(
        kps, pose_seq.frame_size, char_size, segmentation.mask
    )

    # --- Layer 1: skeleton keypoint tracks ----------------------------------
    layer1 = _layer1_skeleton_tracks(kps, body_transform, char_size)
    n1 = layer1.shape[1]  # skeleton anchor count — always preserved in final mix
    # layer1: (F, N1, 2) normalised [0,1]

    # --- Layer 2: Gaussian-falloff interpolated tracks ----------------------
    layer2 = _layer2_interpolated_tracks(layer1, segmentation.mask, density)
    # layer2: (F, N2, 2)

    # --- Layer 3: optical-flow tracks for non-rigid regions -----------------
    layer3, flow_fields = _layer3_flow_tracks(
        stabilized_frames,
        video_body_mask,
        body_transform,
        char_size,
        density,
        flow_estimator=cfg.flow_estimator,
        device=cfg.device,
    )
    # layer3: (F, N3, 2), flow_fields: (F-1, H, W, 2)

    # --- Compose layers; always preserve Layer-1 skeleton anchors -----------
    # Layer-1 tracks are the spec's "scaffold" — they must survive subsampling.
    other_tracks = np.concatenate([layer2, layer3], axis=1)  # (F, N2+N3, 2)
    total_others = other_tracks.shape[1]
    remaining_budget = density - n1

    if remaining_budget <= 0:
        # More skeleton anchors than density budget — keep first `density` only.
        all_tracks = layer1[:, :density, :]
    elif total_others <= remaining_budget:
        # All layers fit within density; pad if still short.
        all_tracks = np.concatenate([layer1, other_tracks], axis=1)
        if all_tracks.shape[1] < density:
            shortage = density - all_tracks.shape[1]
            rng = np.random.default_rng(1)
            pad_idx = rng.choice(all_tracks.shape[1], shortage, replace=True)
            jitter = rng.normal(0, 0.001, (num_frames, shortage, 2)).astype(np.float32)
            pad = np.clip(all_tracks[:, pad_idx, :] + jitter, 0.0, 1.0)
            all_tracks = np.concatenate([all_tracks, pad], axis=1)
    else:
        # Fill remaining budget by randomly sampling from Layer-2+3.
        rng = np.random.default_rng(0)
        idx = rng.choice(total_others, remaining_budget, replace=False)
        all_tracks = np.concatenate([layer1, other_tracks[:, idx, :]], axis=1)

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


def _build_video_body_mask(
    keypoints_f0: np.ndarray,  # (133, 3)
    frame_hw: tuple[int, int],  # (H, W)
) -> np.ndarray:
    """Build a body bounding-box mask in video-frame pixel space from frame-0 pose.

    Uses COCO body keypoints (indices 0–16) to locate the person, adds a ~10%
    margin around the detected bounding box. Falls back to the centre-60% region
    if no confident keypoints are found.

    This mask should be used for camera stabilization and Layer-3 seed selection
    so that those operations work in the correct (video-frame) coordinate space.
    """
    h, w = frame_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    body_kps = keypoints_f0[_BODY_KP_INDICES]  # (17, 3)
    valid = body_kps[body_kps[:, 2] > 0.3]

    if len(valid) >= 2:
        margin_x = max(int(w * 0.10), 5)
        margin_y = max(int(h * 0.10), 5)
        x1 = max(0, int(valid[:, 0].min()) - margin_x)
        y1 = max(0, int(valid[:, 1].min()) - margin_y)
        x2 = min(w, int(valid[:, 0].max()) + margin_x)
        y2 = min(h, int(valid[:, 1].max()) + margin_y)
        mask[y1:y2, x1:x2] = 255
    else:
        # No confident body keypoints: mark the centre 60% as foreground.
        mask[h // 5: 4 * h // 5, w // 5: 4 * w // 5] = 255

    return mask


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
    char_mask: np.ndarray,
) -> np.ndarray:
    """Build a 3×3 similarity transform (uniform scale + translate) mapping
    reference body bounding box → character-image body bounding box.

    Uses the actual character segmentation mask to determine the target body
    region, and a single uniform scale factor to preserve aspect ratio (as
    specified in the design doc).  Falls back gracefully when keypoints or
    mask data are sparse.
    """
    ref_w, ref_h = ref_frame_size
    char_w, char_h = char_image_size

    # Gather confident body keypoints across all frames
    body_kps = keypoints[:, _BODY_KP_INDICES, :]  # (F, 17, 3)
    conf = body_kps[:, :, 2]
    valid_mask = conf > 0.3  # (F, 17)

    if valid_mask.sum() < 2:
        # Fallback: scale entire frame to char image, centred
        scale = min(char_w / max(ref_w, 1), char_h / max(ref_h, 1))
        tx = (char_w - ref_w * scale) / 2
        ty = (char_h - ref_h * scale) / 2
        return np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]], dtype=np.float64)

    xs = body_kps[:, :, 0][valid_mask]
    ys = body_kps[:, :, 1][valid_mask]

    ref_x1, ref_x2 = xs.min(), xs.max()
    ref_y1, ref_y2 = ys.min(), ys.max()
    ref_bw = max(ref_x2 - ref_x1, 1.0)
    ref_bh = max(ref_y2 - ref_y1, 1.0)
    ref_cx = (ref_x1 + ref_x2) / 2
    ref_cy = (ref_y1 + ref_y2) / 2

    # Character body bounds from the segmentation mask bounding box
    char_ys, char_xs = np.where(char_mask > 127)
    if len(char_ys) >= 2:
        c_x1, c_x2 = int(char_xs.min()), int(char_xs.max())
        c_y1, c_y2 = int(char_ys.min()), int(char_ys.max())
        char_bw = max(c_x2 - c_x1, 1)
        char_bh = max(c_y2 - c_y1, 1)
        char_cx = (c_x1 + c_x2) / 2.0
        char_cy = (c_y1 + c_y2) / 2.0
    else:
        # Fallback: 80% centred region
        char_cx = char_w / 2.0
        char_cy = char_h / 2.0
        char_bw = char_w * 0.8
        char_bh = char_h * 0.8

    # Uniform scale: use the smaller of the two axis ratios to avoid clipping
    scale = min(char_bw / ref_bw, char_bh / ref_bh)
    tx = char_cx - ref_cx * scale
    ty = char_cy - ref_cy * scale

    return np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]], dtype=np.float64)


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
    """Return (F, N2, 2) normalised tracks via Gaussian-falloff interpolation.

    Vectorized: computes weighted displacements for all seed points in one
    NumPy operation per frame instead of iterating over individual seeds.
    Seeds are positioned inside the character segmentation mask; their motion
    is a Gaussian-weighted sum of Layer-1 displacements (σ=0.15 in normalised
    space), so they follow skeleton tracks with strength proportional to proximity.
    """
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

    # Frame-0 Layer-1 positions used as displacement anchors
    anchor = layer1_tracks[0]  # (N1, 2)

    # Pre-compute per-seed Gaussian weights once (they depend only on seed positions)
    # diff[i, j] = anchor[j] - seeds[i]  →  shape (actual_n2, N1, 2)
    diff = anchor[np.newaxis, :, :] - seeds[:, np.newaxis, :]
    dist2 = (diff ** 2).sum(axis=2)                                # (actual_n2, N1)
    weights = np.exp(-dist2 / (2 * _GAUSSIAN_SIGMA ** 2))         # (actual_n2, N1)
    w_sum = weights.sum(axis=1, keepdims=True)                     # (actual_n2, 1)
    no_influence = (w_sum.squeeze(1) < 1e-12)                      # (actual_n2,)
    safe_w_sum = np.where(w_sum < 1e-12, np.ones_like(w_sum), w_sum)
    norm_weights = weights / safe_w_sum                            # (actual_n2, N1)

    tracks = np.zeros((num_frames, actual_n2, 2), dtype=np.float32)
    tracks[0] = seeds

    for f in range(1, num_frames):
        delta = layer1_tracks[f] - anchor  # (N1, 2) frame displacement
        # displacement[i] = Σ_j norm_weights[i,j] * delta[j]
        displacement = (norm_weights[:, :, np.newaxis] * delta[np.newaxis, :, :]).sum(axis=1)
        displacement[no_influence] = 0.0
        tracks[f] = np.clip(seeds + displacement, 0.0, 1.0)

    return tracks.astype(np.float32)


# ── Optical flow helpers ──────────────────────────────────────────────────────


def _compute_flow_farneback(
    frame0_bgr: np.ndarray,
    framek_bgr: np.ndarray,
) -> np.ndarray:
    """Dense optical flow via cv2.calcOpticalFlowFarneback (CPU, no deps)."""
    ref_gray = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(framek_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(ref_gray, cur_gray, None, **_FB_PARAMS).astype(np.float32)


def _load_raft(device: str) -> object:
    """Load RAFT-Large from torchvision, cached by device.

    Weights auto-download to ~/.cache/torch/hub/checkpoints/ (~20 MB).
    """
    if device in _raft_cache:
        return _raft_cache[device]

    try:
        import torch  # type: ignore[import]
        from torchvision.models.optical_flow import (  # type: ignore[import]
            Raft_Large_Weights,
            raft_large,
        )
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for RAFT flow estimation. "
            "Run: pip install torchvision"
        ) from exc

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model = model.eval().to(device)
    _raft_cache[device] = model
    return model


def _compute_flow_raft(
    frame0_bgr: np.ndarray,
    framek_bgr: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Dense optical flow via RAFT-Large (GPU recommended, CPU fallback).

    Returns (H, W, 2) float32 flow field in pixels.
    """
    import torch  # type: ignore[import]
    from torchvision.models.optical_flow import Raft_Large_Weights  # type: ignore[import]

    model = _load_raft(device)
    transforms = Raft_Large_Weights.DEFAULT.transforms()

    def _to_tensor(bgr: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W) uint8

    t0 = _to_tensor(frame0_bgr)
    tk = _to_tensor(framek_bgr)
    t0, tk = transforms(t0, tk)           # normalise to float in expected range
    t0 = t0.unsqueeze(0).to(device)       # (1, 3, H, W)
    tk = tk.unsqueeze(0).to(device)

    with torch.no_grad():
        flows = model(t0, tk)             # list of (1, 2, H, W)

    flow = flows[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
    return flow.astype(np.float32)


def _compute_flow_pair(
    frame0_bgr: np.ndarray,
    framek_bgr: np.ndarray,
    estimator: str = "farneback",
    device: str = "cpu",
) -> np.ndarray:
    """Dispatch optical flow computation to the selected estimator.

    Parameters
    ----------
    frame0_bgr:
        Reference frame (BGR uint8).
    framek_bgr:
        Target frame (BGR uint8).
    estimator:
        ``"farneback"`` (CPU, always available) or ``"raft"`` (GPU preferred,
        requires torchvision).  Falls back to Farneback only on ImportError
        (missing dependency) — runtime errors from RAFT are propagated.
    device:
        Torch device string, used only for RAFT.

    Returns
    -------
    np.ndarray
        Dense flow field, shape ``(H, W, 2)`` float32.
    """
    if estimator == "raft":
        try:
            return _compute_flow_raft(frame0_bgr, framek_bgr, device=device)
        except ImportError as exc:
            warnings.warn(
                f"RAFT flow estimation unavailable ({exc}); falling back to Farneback.",
                UserWarning,
                stacklevel=4,
            )
    return _compute_flow_farneback(frame0_bgr, framek_bgr)


def _layer3_flow_tracks(
    stabilized_frames: list[np.ndarray],  # list of (H, W, 3) BGR
    subject_mask: np.ndarray,             # (H, W) uint8 — in video frame space
    body_transform: np.ndarray,           # 3x3
    char_size: tuple[int, int],
    density: int,
    flow_estimator: str = "farneback",
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((F, N3, 2), (F-1, H, W, 2)) optical-flow tracks for non-rigid regions.

    Flow is computed frame-0 → frame-k directly (never chained).
    ``subject_mask`` must be in video-frame pixel space (same size as the frames).
    """
    num_frames = len(stabilized_frames)
    n3 = density // 4

    fh, fw = stabilized_frames[0].shape[:2]

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
        # Frame-0 → frame-k directly (spec requirement — never chain)
        flow = _compute_flow_pair(
            stabilized_frames[0], stabilized_frames[k],
            estimator=flow_estimator, device=device,
        )
        flow_fields.append(flow)  # (H, W, 2)

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
