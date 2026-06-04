"""ComfyUI model-management integration helpers.

This module is intentionally small. Generation must enter through here so future
Wan/Concat-ID model loading can cooperate with ComfyUI VRAM arbitration instead
of bypassing it with direct CUDA allocation.
"""
from __future__ import annotations

from pathlib import Path


def get_comfy_model_management():
    try:
        import comfy.model_management as model_management  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Motion Mirror ComfyUI nodes must run inside a ComfyUI environment."
        ) from exc
    return model_management


def throw_if_interrupted() -> None:
    model_management = get_comfy_model_management()
    hook = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if hook is not None:
        hook()


def run_motion_mirror_generation(
    *,
    image_path: str,
    motion_video_path: str,
    backend: str,
    resolution: str,
    frames: int,
    density: int,
    device: str,
) -> tuple[str]:
    throw_if_interrupted()
    from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline

    config = MotionMirrorConfig(
        backend=backend,
        resolution=resolution,
        num_frames=int(frames),
        trajectory_density=int(density),
        device=device,
    )
    result = MotionMirrorPipeline(config).run(
        Path(image_path),
        Path(motion_video_path),
    )
    throw_if_interrupted()
    return (str(result.output_path),)
