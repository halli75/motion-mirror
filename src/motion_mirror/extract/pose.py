from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import MotionMirrorConfig
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
    ValueError
        If the file extension is unsupported or the video cannot be opened.
    RuntimeError
        If the video contains no readable frames.
    """
    cfg = config or MotionMirrorConfig()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported video format {video_path.suffix!r}. "
            f"Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from: {video_path}")

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
    # ENGINEER NOTE: rtmlib class names vary by version.
    # Verify with: pip show rtmlib && python -c "import rtmlib; print(dir(rtmlib))"
    # before relying on these imports.
    try:
        from rtmlib import PoseTracker  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "rtmlib is not installed. Run: pip install -r requirements-cuda.txt"
        ) from exc

    pose_model_path = cfg.model_cache("dwpose") / "rtmpose-l_8xb32-270e_coco-wholebody-384x288-eaf3f731_20230312.onnx"
    det_model_path = cfg.model_cache("dwpose") / "yolox_l_8xb8-300e_humanart-a39d44ed.onnx"

    for p in (pose_model_path, det_model_path):
        if not p.exists():
            raise FileNotFoundError(
                f"DWPose model not found: {p}\n"
                "Run: motion-mirror download --model dwpose"
            )

    backend_ep = "onnxruntime" if cfg.device == "cuda" else "cpu"
    tracker = PoseTracker(
        det=det_model_path,
        pose=pose_model_path,
        backend=backend_ep,
        tracking=False,
    )

    keypoints_list: list[np.ndarray] = []
    for frame in frames:
        kps, _ = tracker(frame)  # (133, 3)
        keypoints_list.append(kps.astype(np.float32))

    keypoints = np.stack(keypoints_list)  # (F, 133, 3)

    return PoseSequence(
        source_video_path=video_path,
        keypoints=keypoints,
        frame_size=(frame_w, frame_h),
        fps=fps,
    )
