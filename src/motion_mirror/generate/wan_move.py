"""Wan-Move-14B generation backend.

Mock path:
    When config.backend == 'mock', writes a solid-colour MP4 using
    cv2.VideoWriter.  No GPU or model weights required.  Used for all
    CPU-only tests and for end-to-end pipeline smoke runs.

Real path (v0.1 stub — requires GPU + downloaded weights):
    Loads Wan-Move-14B via huggingface_hub and calls the model.

    ENGINEER NOTE before implementing the real path:
    Verify the exact Python package/import for Wan-Move.  As of the plan
    date the HuggingFace repo is Wan-AI/Wan2.1-I2V-14B-720P but the
    installable package name (wan, wan_video, wanvideo …) is unconfirmed.
    Run: pip index versions wan  (or search PyPI / the model card README).
    This is a hard blocker — do not assume the import path.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest


def generate_with_wan_move(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate an animated video from a segmented character image + trajectory map.

    Parameters
    ----------
    request:
        Fully populated ``GenerationRequest`` (paths, backend, resolution, …).
    config:
        Pipeline config.  Defaults created if omitted.

    Returns
    -------
    GenerationResult
        Points to the generated video on disk.

    Raises
    ------
    FileNotFoundError
        Real path only — if model weights are missing.
    ImportError
        Real path only — if the wan package is not installed.
    ValueError
        If the resolution string is malformed.
    """
    cfg = config or MotionMirrorConfig()

    # Parse resolution
    try:
        w_str, h_str = request.resolution.split("x")
        out_w, out_h = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {request.resolution!r}. Expected 'WxH', e.g. '832x480'."
        ) from exc

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Mock path ─────────────────────────────────────────────────────────────
    if cfg.backend == "mock" or request.backend == "mock":
        return _generate_mock(request, out_w, out_h)

    # ── Real path ─────────────────────────────────────────────────────────────
    return _generate_real(request, cfg, out_w, out_h)


# ── Mock implementation ────────────────────────────────────────────────────────


def _generate_mock(
    request: GenerationRequest,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Write a solid-colour MP4 of the requested size and frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(request.output_path), fourcc, 16.0, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open output path: {request.output_path}"
        )

    # Deterministic colour derived from the seed so tests can assert it
    rng = np.random.default_rng(request.seed)
    colour = rng.integers(50, 200, size=3, dtype=np.uint8).tolist()
    frame = np.full((out_h, out_w, 3), colour, dtype=np.uint8)

    for _ in range(request.frames):
        writer.write(frame)
    writer.release()

    return GenerationResult(
        video_path=request.output_path,
        backend="mock",
        resolution=request.resolution,
        num_frames=request.frames,
    )


# ── Real implementation (v0.1 stub) ───────────────────────────────────────────


def _generate_real(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Invoke Wan-Move-14B to generate the output video.

    This is a v0.1 stub.  The model import path must be verified before
    this function can be completed (see module docstring).
    """
    # Validate model weights exist before touching GPU
    model_dir = config.model_cache("wan-move")
    if not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Wan-Move-14B weights not found in {model_dir}.\n"
            "Run: motion-mirror download --model wan-move"
        )

    # Validate input files exist
    for label, p in [
        ("segmented image", request.segmented_image_path),
        ("trajectory map", request.trajectory_map_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Generation input {label} not found: {p}")

    try:
        import torch  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "torch is not installed. Run: pip install -r requirements-cuda.txt"
        ) from exc

    # Free any cached VRAM before loading the large model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── TODO: replace with verified import once package name is confirmed ──
    # Expected interface (unverified):
    #   from wan import WanMove          # verify package name
    #   model = WanMove.from_pretrained(str(model_dir), device=config.device)
    #   model.generate(
    #       image=request.segmented_image_path,
    #       trajectory=request.trajectory_map_path,
    #       output=request.output_path,
    #       width=out_w, height=out_h,
    #       num_frames=request.frames,
    #       seed=request.seed,
    #   )
    raise NotImplementedError(
        "Wan-Move-14B real path is not yet implemented.\n"
        "Verify the Python package name from the model card, then complete "
        "_generate_real() in generate/wan_move.py.\n"
        "Use --backend mock for testing until then."
    )
