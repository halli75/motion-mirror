"""Character segmentation — Stage 1 of the Motion Mirror pipeline.

Two backends are supported:

  rembg (default) — u2net model, pip-installable, <1 s/image, no GPU needed.
  sam2            — SAM-2 Large, higher quality on complex scenes, GPU recommended.
                    Requires: pip install git+https://github.com/facebookresearch/sam2.git
                    Model auto-downloads from HuggingFace on first use (~900 MB).

The active backend is controlled by ``MotionMirrorConfig.segmenter``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..config import MotionMirrorConfig
from ..exceptions import UnsupportedImageError
from ..types import SegmentationResult

_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# ── rembg singleton ───────────────────────────────────────────────────────────

# Module-level lazy session — expensive to construct, reused across calls.
_rembg_session: object | None = None


def _get_rembg_session() -> object:
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


# ── SAM-2 singleton ───────────────────────────────────────────────────────────

# Cached per-device so we don't reload if the same device is requested again.
_sam2_predictors: dict[str, object] = {}


def _get_sam2_predictor(device: str = "cuda") -> object:
    """Lazy-load SAM-2 Large from HuggingFace, cached by device."""
    if device in _sam2_predictors:
        return _sam2_predictors[device]

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "SAM-2 is not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Then download the weights:\n"
            "  motion-mirror download --model sam2"
        ) from exc

    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large",
        device=device,
    )
    _sam2_predictors[device] = predictor
    return predictor


# ── Public API ────────────────────────────────────────────────────────────────


def segment_subject(
    image_path: Path,
    config: MotionMirrorConfig | None = None,
) -> SegmentationResult:
    """Remove background from *image_path* and return a SegmentationResult.

    The RGBA output is saved to ``config.output_dir / "segmented.png"``.

    Parameters
    ----------
    image_path:
        Path to the character image.
    config:
        Pipeline configuration.  ``config.segmenter`` selects the backend
        (``"rembg"`` or ``"sam2"``).

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    UnsupportedImageError
        If the file extension is not supported.
    ImportError
        If ``segmenter="sam2"`` but SAM-2 is not installed.
    """
    cfg = config or MotionMirrorConfig()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise UnsupportedImageError(
            f"Unsupported image format {image_path.suffix!r}. "
            f"Accepted formats: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    if cfg.segmenter == "sam2":
        return _segment_sam2(image_path, cfg)
    return _segment_rembg(image_path, cfg)


# ── rembg backend ─────────────────────────────────────────────────────────────


def _segment_rembg(
    image_path: Path,
    cfg: MotionMirrorConfig,
) -> SegmentationResult:
    """Background removal via rembg u2net (CPU, no GPU required)."""
    from PIL import Image
    from rembg import remove

    input_image = Image.open(image_path).convert("RGBA")
    rgba_pil: Image.Image = remove(input_image, session=_get_rembg_session())

    rgba_np = np.array(rgba_pil, dtype=np.uint8)        # (H, W, 4)
    mask_np = rgba_np[:, :, 3]                           # alpha channel
    mask_np = np.where(mask_np > 0, np.uint8(255), np.uint8(0))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = cfg.output_dir / "segmented.png"
    rgba_pil.save(str(rgba_path))

    return SegmentationResult(
        source_image_path=image_path,
        rgba_path=rgba_path,
        mask=mask_np,
        rgba=rgba_np,
    )


# ── SAM-2 backend ─────────────────────────────────────────────────────────────


def _segment_sam2(
    image_path: Path,
    cfg: MotionMirrorConfig,
) -> SegmentationResult:
    """Background removal via SAM-2 Large with a centre-point prompt.

    SAM-2 is prompted with the image centre (assumed to contain the character).
    ``multimask_output=True`` generates three candidate masks; the one with the
    highest confidence score is selected.

    For images where the character is not centred, pass an explicit bounding-box
    or point prompt via the Python API directly.

    Requires
    --------
    pip install git+https://github.com/facebookresearch/sam2.git
    """
    from PIL import Image

    device = cfg.device if cfg.device == "cuda" and _cuda_available() else "cpu"
    predictor = _get_sam2_predictor(device)
    import torch  # type: ignore[import]  # only needed for inference context

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img, dtype=np.uint8)       # (H, W, 3) RGB
    h, w = img_np.shape[:2]

    # Centre-point prompt — assumes the character occupies the image centre
    cx, cy = w // 2, h // 2
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)      # 1 = foreground

    inference_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.inference_mode()
    )

    with torch.inference_mode(), inference_ctx:
        predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,      # returns 3 candidates
        )

    # Pick the mask with the highest SAM-2 confidence score
    best_idx = int(scores.argmax())
    mask_bool = masks[best_idx]                        # (H, W) bool
    mask_np = (mask_bool.astype(np.uint8) * 255)      # (H, W) uint8

    # Warn if the chosen mask is very small or very large
    frac = mask_bool.sum() / (h * w)
    if frac < 0.05:
        warnings.warn(
            f"SAM-2 mask covers only {frac:.1%} of the image — the character "
            "may not be centred. Consider cropping the image first.",
            UserWarning,
            stacklevel=3,
        )
    elif frac > 0.85:
        warnings.warn(
            f"SAM-2 mask covers {frac:.1%} of the image — this may be the "
            "background rather than the character.",
            UserWarning,
            stacklevel=3,
        )

    # Build RGBA with mask as alpha channel
    rgba_np = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_np[:, :, :3] = img_np
    rgba_np[:, :, 3] = mask_np

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = cfg.output_dir / "segmented.png"
    Image.fromarray(rgba_np, "RGBA").save(str(rgba_path))

    return SegmentationResult(
        source_image_path=image_path,
        rgba_path=rgba_path,
        mask=mask_np,
        rgba=rgba_np,
    )


def _cuda_available() -> bool:
    """Return True if a CUDA GPU is available, without raising."""
    try:
        import torch  # type: ignore[import]
        return torch.cuda.is_available()
    except Exception:
        return False
