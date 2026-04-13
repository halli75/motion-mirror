"""ControlNet-OpenPose generation backend (lightweight fallback).

v0.1 status: stub only.  The real implementation (Wan2.1-1.3B +
ControlNet-OpenPose) is planned for v0.2.  The mock path is provided
for end-to-end pipeline testing without a GPU.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest


def generate_with_controlnet(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate via the lightweight ControlNet-OpenPose backend.

    Parameters
    ----------
    request:
        Fully populated ``GenerationRequest``.
    config:
        Pipeline config.  Defaults created if omitted.

    Returns
    -------
    GenerationResult
        Points to the generated video on disk.
    """
    cfg = config or MotionMirrorConfig()

    try:
        w_str, h_str = request.resolution.split("x")
        out_w, out_h = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {request.resolution!r}. Expected 'WxH'."
        ) from exc

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "mock" or request.backend == "mock":
        return _generate_mock(request, out_w, out_h)

    raise NotImplementedError(
        "ControlNet real path is planned for v0.2. Use --backend mock for now."
    )


def _generate_mock(
    request: GenerationRequest,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(request.output_path), fourcc, 16.0, (out_w, out_h)
    )
    rng = np.random.default_rng(request.seed + 1)
    colour = rng.integers(50, 200, size=3, dtype=np.uint8).tolist()
    frame = np.full((out_h, out_w, 3), colour, dtype=np.uint8)
    for _ in range(request.frames):
        writer.write(frame)
    writer.release()
    return GenerationResult(
        video_path=request.output_path,
        backend="mock-controlnet",
        resolution=request.resolution,
        num_frames=request.frames,
    )
