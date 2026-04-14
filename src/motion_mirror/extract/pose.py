from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import MotionMirrorConfig
from ..exceptions import (
    MultiplePeopleDetectedError,
    NoPoseDetectedError,
    SmallSubjectError,
    SmallSubjectWarning,
    UnsupportedVideoError,
    VideoDecodeError,
)
from ..types import PoseSequence

_SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def extract_pose(
    video_path: Path,
    config: MotionMirrorConfig | None = None,
) -> PoseSequence:
    """Extract per-frame pose keypoints from *video_path*.

    Uses DWPose-L (via rtmlib) when model weights are available and
    ``config.backend != 'mock'``.  In mock mode random keypoints of the
    correct shape are returned so the rest of the pipeline can be tested
    without a GPU or downloaded models.

    Keypoints layout: COCO-WholeBody — 133 keypoints, axis-2 = [x_px, y_px, conf].

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    UnsupportedVideoError
        If the file extension is not supported.
    VideoDecodeError
        If OpenCV cannot open or read frames from the video.
    NoPoseDetectedError
        If DWPose finds no person in the video (real path only).
    MultiplePeopleDetectedError
        If DWPose finds more than one person (real path only).
    SmallSubjectError
        If the detected person occupies < 5 % of the frame area (real path only).
    """
    import warnings

    cfg = config or MotionMirrorConfig()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise UnsupportedVideoError(
            f"Unsupported video format {video_path.suffix!r}. "
            f"Accepted formats: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Warn on extreme frame rates (temporal mapping quality degrades)
    if fps < 12.0 or fps > 60.0:
        import warnings as _w
        _w.warn(
            f"Reference video FPS is {fps:.1f} — outside the 12–60 fps range. "
            "Temporal mapping quality may be degraded.",
            UserWarning,
            stacklevel=2,
        )

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise VideoDecodeError(f"No frames could be read from: {video_path}")

    # --- Mock path: return random keypoints of correct shape (no model needed) ---
    if cfg.backend == "mock":
        rng = np.random.default_rng(0)
        kps = rng.random((len(frames), 133, 3)).astype(np.float32)
        # Scale x/y to frame dimensions; confidence in [0.5, 1.0] so they look real
        kps[:, :, 0] *= frame_w
        kps[:, :, 1] *= frame_h
        kps[:, :, 2] = kps[:, :, 2] * 0.5 + 0.5
        return PoseSequence(
            source_video_path=video_path,
            keypoints=kps,
            frame_size=(frame_w, frame_h),
            fps=fps,
        )

    # --- Real path: DWPose-L via rtmlib ---
    # rtmlib >=0.0.13 exposes Wholebody for 133-keypoint COCO-WholeBody estimation.
    # Older versions used PoseTracker(det=..., pose=...) which was removed.
    try:
        import rtmlib as _rtmlib  # type: ignore[import]
        _Wholebody = getattr(_rtmlib, "Wholebody", None)
        _PoseTracker = getattr(_rtmlib, "PoseTracker", None)
        if _Wholebody is None and _PoseTracker is None:
            raise ImportError("rtmlib: neither Wholebody nor PoseTracker found")
    except ImportError as exc:
        raise ImportError(
            "rtmlib is not installed. Run: pip install -r requirements-cuda.txt"
        ) from exc

    pose_model_path = cfg.model_cache("dwpose") / "dw-ll_ucoco_384.onnx"
    det_model_path  = cfg.model_cache("dwpose") / "yolox_l.onnx"

    for p in (pose_model_path, det_model_path):
        if not p.exists():
            raise FileNotFoundError(
                f"DWPose model not found: {p}\n"
                "Run: motion-mirror download --model dwpose"
            )

    backend_ep = "onnxruntime" if cfg.device == "cuda" else "cpu"

    # Try Wholebody first (current API), fall back to legacy PoseTracker
    tracker = None
    use_wholebody = False
    if _Wholebody is not None:
        try:
            tracker = _Wholebody(
                det=str(det_model_path),
                pose=str(pose_model_path),
                to_openpose=False,
                backend=backend_ep,
                device=cfg.device,
            )
            use_wholebody = True
        except TypeError:
            pass  # fall through to PoseTracker

    if tracker is None and _PoseTracker is not None:
        tracker = _PoseTracker(
            str(pose_model_path),
            str(det_model_path),
            backend=backend_ep,
        )

    if tracker is None:
        raise RuntimeError("Could not initialise rtmlib tracker — check rtmlib version")

    frame_area = frame_w * frame_h
    keypoints_list: list[np.ndarray] = []

    for frame_idx, frame in enumerate(frames):
        result = tracker(frame)

        # Normalise output to (num_people, 133, 3) float32 [x, y, conf].
        # Wholebody returns (keypoints, scores) where:
        #   keypoints: (N, 133, 2) float32  — x, y pixel coords
        #   scores:    (N, 133)   float32  — confidence per keypoint
        # Some versions already pack conf into dim-2 → (N, 133, 3).
        if isinstance(result, (tuple, list)) and len(result) == 2:
            kps_xy, kps_score = result
            if kps_xy is None or (hasattr(kps_xy, "__len__") and len(kps_xy) == 0):
                kps_raw = np.zeros((0, 133, 3), dtype=np.float32)
            else:
                kps_xy = np.asarray(kps_xy, dtype=np.float32)
                kps_score = np.asarray(kps_score, dtype=np.float32)
                if kps_xy.ndim == 2:
                    kps_xy = kps_xy[np.newaxis]
                    kps_score = kps_score[np.newaxis] if kps_score.ndim == 1 else kps_score
                if kps_xy.shape[-1] == 2:
                    kps_raw = np.concatenate(
                        [kps_xy, kps_score[:, :, np.newaxis]], axis=2
                    )  # (N, 133, 3)
                else:
                    kps_raw = kps_xy  # already (N, 133, 3)
        else:
            kps_raw = np.asarray(result, dtype=np.float32)
            if kps_raw.ndim == 2:
                kps_raw = kps_raw[np.newaxis]

        # kps_raw is now (num_people, 133, 3)
        if kps_raw.ndim == 2:
            kps_raw = kps_raw[np.newaxis]  # add person dim → (1, 133, 3)

        num_people = kps_raw.shape[0]

        # ── Person-count validation (frame 0 only to avoid repeated errors) ──
        if frame_idx == 0:
            if num_people == 0:
                raise NoPoseDetectedError(
                    "No person detected in the reference video. "
                    "Ensure a person is clearly visible in the frame."
                )
            if num_people > 1:
                raise MultiplePeopleDetectedError(
                    f"Multiple people detected ({num_people}) in the reference video. "
                    "Motion Mirror v0.1 supports single-person transfer only. "
                    "Crop the reference video to show one person, or use "
                    "--person-index N to select which person to track.",
                    count=num_people,
                )

            # ── Subject size check (frame 0 only) ────────────────────────────
            # Estimate bounding box from confident keypoints (conf > 0.3)
            conf = kps_raw[0, :, 2]
            xy = kps_raw[0, :, :2][conf > 0.3]
            if xy.size > 0:
                x_min, y_min = xy.min(axis=0)
                x_max, y_max = xy.max(axis=0)
                bbox_area = max(0.0, (x_max - x_min) * (y_max - y_min))
                bbox_fraction = bbox_area / frame_area if frame_area > 0 else 0.0

                if bbox_fraction < 0.05:
                    raise SmallSubjectError(
                        f"Detected person occupies only {bbox_fraction:.1%} of the frame "
                        "(threshold: 5%). Use a closer shot or crop the video.",
                        bbox_fraction=bbox_fraction,
                    )
                if bbox_fraction < 0.10:
                    warnings.warn(
                        f"Detected person occupies {bbox_fraction:.1%} of the frame "
                        "(below 10%). Output quality may be degraded. "
                        "Consider using a closer shot.",
                        SmallSubjectWarning,
                        stacklevel=2,
                    )

        keypoints_list.append(kps_raw[0].astype(np.float32))  # take person 0

    keypoints = np.stack(keypoints_list)  # (F, 133, 3)

    return PoseSequence(
        source_video_path=video_path,
        keypoints=keypoints,
        frame_size=(frame_w, frame_h),
        fps=fps,
    )
