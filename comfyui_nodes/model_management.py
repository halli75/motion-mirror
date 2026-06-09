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


def maybe_throw_if_interrupted() -> None:
    """Like throw_if_interrupted, but a no-op outside a ComfyUI environment.

    The extraction nodes are CPU-capable and useful standalone, so they only
    honour the interrupt hook when ComfyUI is actually present.
    """
    try:
        throw_if_interrupted()
    except RuntimeError:
        pass


def run_motion_mirror_generation(
    *,
    image_path: str,
    motion_video_path: str,
    backend: str,
    resolution: str,
    frames: int,
    density: int,
    device: str,
    pose_path: str = "",
    trajectory_path: str = "",
) -> tuple[str]:
    throw_if_interrupted()
    from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline
    from motion_mirror.types import PoseSequence, TrajectoryMap

    pose = PoseSequence.load(Path(pose_path)) if pose_path else None
    trajectory = TrajectoryMap.load(Path(trajectory_path)) if trajectory_path else None

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
        pose=pose,
        trajectory=trajectory,
    )
    throw_if_interrupted()
    return (str(result.output_path),)
