from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GenerationRequest:
    """Inputs to a generation backend.

    segmented_image_path - RGBA PNG output from segment_subject()
    trajectory_map_path  - .npz file saved from TrajectoryMap.save()
    output_path          - where the generated video should be written
    """

    segmented_image_path: Path
    trajectory_map_path: Path
    output_path: Path
    conditioning_video_path: Path | None = None
    conditioning_mask_path: Path | None = None

    backend: str = "wan-move-14b"
    resolution: str = "832x480"
    frames: int = 81
    device: str = "cuda"
    seed: int = 0
