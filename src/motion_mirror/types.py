from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SegmentationResult:
    """Output of segment_subject().

    mask  — (H, W) uint8, 0 = background, 255 = foreground
    rgba  — (H, W, 4) uint8, alpha channel matches mask
    """

    source_image_path: Path
    rgba_path: Path
    mask: np.ndarray  # (H, W) uint8
    rgba: np.ndarray  # (H, W, 4) uint8


@dataclass(slots=True)
class PoseSequence:
    """Output of extract_pose().

    keypoints — (F, 133, 3) float32
      axis 2: [x_px, y_px, confidence]
      133 keypoints = COCO-WholeBody layout (DWPose-L output)
    frame_size — (W, H) of source video frames
    """

    source_video_path: Path
    keypoints: np.ndarray  # (F, 133, 3) float32
    frame_size: tuple[int, int]  # (W, H)
    fps: float


@dataclass(slots=True)
class TrajectoryMap:
    """Output of synthesize_trajectory().

    tracks      — (F, density, 2) float32, values in [0, 1] character-image space
    flow_fields — (F-1, H, W, 2) float32, dense optical flow per frame pair
    frame_size  — (W, H) character-image space
    """

    tracks: np.ndarray  # (F, density, 2) float32
    flow_fields: np.ndarray  # (F-1, H, W, 2) float32
    density: int
    frame_size: tuple[int, int]  # (W, H) character-image space

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            tracks=self.tracks,
            flow_fields=self.flow_fields,
            density=np.array(self.density),
            frame_w=np.array(self.frame_size[0]),
            frame_h=np.array(self.frame_size[1]),
        )

    @classmethod
    def load(cls, path: Path) -> TrajectoryMap:
        data = np.load(path)
        return cls(
            tracks=data["tracks"],
            flow_fields=data["flow_fields"],
            density=int(data["density"]),
            frame_size=(int(data["frame_w"]), int(data["frame_h"])),
        )


@dataclass(slots=True)
class GenerationResult:
    """Output of generate_with_wan_move() / generate_with_controlnet()."""

    video_path: Path
    backend: str
    resolution: str
    num_frames: int
