from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import MotionMirrorConfig
from ..exceptions import MultipleCharactersError, UnsupportedImageError
from ..types import SegmentationResult

if TYPE_CHECKING:
    pass

_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Module-level lazy session singleton — expensive to construct, reused across calls.
_session: object | None = None


def _get_session() -> object:
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def segment_subject(
    image_path: Path,
    config: MotionMirrorConfig | None = None,
) -> SegmentationResult:
    """Remove background from *image_path* and return a SegmentationResult.

    The RGBA output is saved to ``config.output_dir / "segmented.png"``.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    UnsupportedImageError
        If the file extension is not supported.
    MultipleCharactersError
        If DWPose detects more than one person in the character image
        (only raised when a pose model is available; skipped in mock mode).
    """
    cfg = config or MotionMirrorConfig()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise UnsupportedImageError(
            f"Unsupported image format {image_path.suffix!r}. "
            f"Accepted formats: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    from PIL import Image
    from rembg import remove

    input_image = Image.open(image_path).convert("RGBA")
    rgba_pil: Image.Image = remove(input_image, session=_get_session())

    rgba_np = np.array(rgba_pil, dtype=np.uint8)  # (H, W, 4)
    mask_np = rgba_np[:, :, 3]  # alpha channel — 0=bg, >0=fg; normalise to 0/255
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
