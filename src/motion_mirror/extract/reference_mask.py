"""Reference-video subject masks for trajectory and VACE conditioning."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
import warnings

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import PoseSequence, ReferenceMaskResult

_BODY_KP_INDICES = list(range(17))
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"
_sam2_video_predictors: dict[str, object] = {}


@dataclass(slots=True)
class _InitialPrompt:
    points: np.ndarray | None
    labels: np.ndarray | None
    box: np.ndarray | None


def propagate_reference_masks(
    video_path: Path,
    pose_seq: PoseSequence,
    config: MotionMirrorConfig | None = None,
) -> ReferenceMaskResult:
    """Propagate a reference-video subject mask with SAM-2."""
    cfg = config or MotionMirrorConfig()
    if not video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {video_path}")

    frames, fps, frame_size = _load_video_frames(video_path)
    if not frames:
        raise ValueError(f"Reference video produced no readable frames: {video_path}")

    frame_w, frame_h = frame_size
    device = cfg.device if cfg.device == "cuda" and _cuda_available() else "cpu"
    predictor = _get_sam2_video_predictor(device)
    prompt = _build_initial_prompt(pose_seq, frame_size)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    masks = np.zeros((len(frames), frame_h, frame_w), dtype=np.uint8)
    seen = np.zeros(len(frames), dtype=bool)

    with tempfile.TemporaryDirectory(prefix="motion_mirror_sam2_") as tmp:
        frame_dir = Path(tmp)
        _write_sam2_frame_dir(frames, frame_dir)

        with _sam2_inference_context(device):
            state = predictor.init_state(video_path=str(frame_dir))  # type: ignore[attr-defined]
            response = _add_initial_prompt(predictor, state, prompt)
            seed_mask = _mask_from_predictor_response(response, (frame_h, frame_w))
            if seed_mask is not None:
                masks[0] = seed_mask
                seen[0] = True

            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):  # type: ignore[attr-defined]
                idx = int(frame_idx)
                if idx < 0 or idx >= len(frames):
                    continue
                masks[idx] = _mask_logits_to_uint8(mask_logits, (frame_h, frame_w))
                seen[idx] = True

    if not seen.any():
        raise RuntimeError("SAM-2 did not return any propagated reference masks.")

    _fill_missing_masks_in_place(masks, seen)
    _warn_on_extreme_masks(masks)

    mask_video_path = cfg.output_dir / "reference_mask.mp4"
    _write_mask_video(mask_video_path, masks, fps=fps)

    return ReferenceMaskResult(
        source_video_path=video_path,
        mask_video_path=mask_video_path,
        masks=masks,
        frame_size=frame_size,
        fps=fps,
    )


def resample_reference_masks(
    reference_masks: ReferenceMaskResult,
    num_frames: int,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return masks resampled to ``num_frames`` and optionally resized to ``size``."""
    masks = reference_masks.masks
    if masks.shape[0] == 0:
        raise ValueError("ReferenceMaskResult contains no masks")

    if num_frames > 0 and masks.shape[0] != num_frames:
        indices = np.linspace(0, masks.shape[0] - 1, num_frames).round().astype(np.int32)
        masks = masks[indices]

    if size is None or reference_masks.frame_size == size:
        return masks.astype(np.uint8, copy=False)

    out_w, out_h = size
    resized = [
        cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    return np.stack(resized).astype(np.uint8)


def write_vace_reference_mask_video(
    reference_masks: ReferenceMaskResult,
    path: Path,
    size: tuple[int, int],
    num_frames: int,
    fps: float = 16.0,
) -> Path:
    """Write SAM-2 masks in the VACE mask polarity used by this pipeline."""
    masks = resample_reference_masks(reference_masks, num_frames, size=size)
    frames = [_reference_mask_to_vace_frame(mask) for mask in masks]
    _write_bgr_video(path, frames, fps=fps)
    return path


def _get_sam2_video_predictor(device: str = "cuda") -> object:
    if device in _sam2_video_predictors:
        return _sam2_video_predictors[device]

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "SAM-2 video propagation is not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Then download the weights:\n"
            "  motion-mirror download --model sam2"
        ) from exc

    predictor = SAM2VideoPredictor.from_pretrained(_SAM2_MODEL_ID, device=device)
    _sam2_video_predictors[device] = predictor
    return predictor


def _load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 16.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        return [], 16.0, (0, 0)
    h, w = frames[0].shape[:2]
    return frames, fps, (w, h)


def _write_sam2_frame_dir(frames: list[np.ndarray], frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        ok = cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), frame)
        if not ok:
            raise RuntimeError(f"Could not write SAM-2 frame {idx} to {frame_dir}")


def _build_initial_prompt(
    pose_seq: PoseSequence,
    frame_size: tuple[int, int],
) -> _InitialPrompt:
    frame_w, frame_h = frame_size
    if pose_seq.keypoints.shape[0] > 0:
        body = pose_seq.keypoints[0, _BODY_KP_INDICES]
        valid = body[body[:, 2] > 0.3]
        if len(valid) >= 2:
            src_w, src_h = pose_seq.frame_size
            scale_x = frame_w / max(src_w, 1)
            scale_y = frame_h / max(src_h, 1)
            xs = valid[:, 0] * scale_x
            ys = valid[:, 1] * scale_y
            margin_x = max(frame_w * 0.10, 5.0)
            margin_y = max(frame_h * 0.10, 5.0)
            box = np.array(
                [
                    max(0.0, float(xs.min() - margin_x)),
                    max(0.0, float(ys.min() - margin_y)),
                    min(float(frame_w - 1), float(xs.max() + margin_x)),
                    min(float(frame_h - 1), float(ys.max() + margin_y)),
                ],
                dtype=np.float32,
            )
            return _InitialPrompt(points=None, labels=None, box=box)

    point = np.array([[frame_w / 2.0, frame_h / 2.0]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    return _InitialPrompt(points=point, labels=label, box=None)


def _add_initial_prompt(
    predictor: object,
    state: object,
    prompt: _InitialPrompt,
) -> object:
    kwargs: dict[str, object] = {
        "inference_state": state,
        "frame_idx": 0,
        "obj_id": 1,
    }
    if prompt.box is not None:
        kwargs["box"] = prompt.box
    else:
        kwargs["points"] = prompt.points
        kwargs["labels"] = prompt.labels

    try:
        return predictor.add_new_points_or_box(**kwargs)  # type: ignore[attr-defined]
    except TypeError:
        if "points" not in kwargs:
            raise
        kwargs["point_coords"] = kwargs.pop("points")
        kwargs["point_labels"] = kwargs.pop("labels")
        return predictor.add_new_points_or_box(**kwargs)  # type: ignore[attr-defined]


def _mask_from_predictor_response(
    response: object,
    frame_hw: tuple[int, int],
) -> np.ndarray | None:
    if isinstance(response, tuple) and len(response) >= 3:
        return _mask_logits_to_uint8(response[2], frame_hw)
    return None


def _mask_logits_to_uint8(
    mask_logits: object,
    frame_hw: tuple[int, int],
) -> np.ndarray:
    arr = _as_numpy(mask_logits)
    while arr.ndim > 2:
        arr = arr[0]

    frame_h, frame_w = frame_hw
    if arr.shape != (frame_h, frame_w):
        arr = cv2.resize(arr.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    if arr.dtype == np.bool_:
        foreground = arr
    elif float(arr.min(initial=0.0)) >= 0.0 and float(arr.max(initial=0.0)) <= 1.0:
        foreground = arr > 0.5
    else:
        foreground = arr > 0.0
    return (foreground.astype(np.uint8) * 255).astype(np.uint8)


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("SAM-2 returned an empty mask list")
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[attr-defined]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[attr-defined]
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())  # type: ignore[attr-defined]
    return np.asarray(value)


def _fill_missing_masks_in_place(masks: np.ndarray, seen: np.ndarray) -> None:
    last: np.ndarray | None = None
    for idx in range(masks.shape[0]):
        if seen[idx]:
            last = masks[idx].copy()
        elif last is not None:
            masks[idx] = last

    next_mask: np.ndarray | None = None
    for idx in range(masks.shape[0] - 1, -1, -1):
        if seen[idx]:
            next_mask = masks[idx].copy()
        elif next_mask is not None:
            masks[idx] = next_mask


def _warn_on_extreme_masks(masks: np.ndarray) -> None:
    first = masks[0] > 127
    frac = float(first.mean())
    if frac < 0.02:
        warnings.warn(
            f"SAM-2 reference mask covers only {frac:.1%} of frame 0.",
            UserWarning,
            stacklevel=3,
        )
    elif frac > 0.90:
        warnings.warn(
            f"SAM-2 reference mask covers {frac:.1%} of frame 0; it may include background.",
            UserWarning,
            stacklevel=3,
        )


def _write_mask_video(path: Path, masks: np.ndarray, fps: float) -> None:
    frames = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in masks]
    _write_bgr_video(path, frames, fps=fps)


def _reference_mask_to_vace_frame(mask: np.ndarray) -> np.ndarray:
    # VACE mask polarity: white = generate, black = keep from the control
    # video. The subject must be WHITE — a black subject makes VACE copy the
    # control video's skeleton pixels through verbatim (2026-07-04 GPU run).
    vace_mask = np.where(mask > 127, np.uint8(255), np.uint8(0))
    return cv2.cvtColor(vace_mask, cv2.COLOR_GRAY2BGR)


def _write_bgr_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("No frames provided for reference mask video")

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
                raise ValueError("Reference mask frames must all have the same size")
            writer.write(frame)
    finally:
        writer.release()


@contextmanager
def _sam2_inference_context(device: str):
    try:
        import torch  # type: ignore[import]
    except ImportError:
        yield
        return

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import]
        return bool(torch.cuda.is_available())
    except Exception:
        return False
