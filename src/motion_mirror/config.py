from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class MotionMirrorConfig:
    project_root: Path = field(default_factory=Path.cwd)
    output_dir_name: str = "outputs"

    # Trajectory
    trajectory_density: int = 512  # 512 = default, 1024 = HQ

    # Generation
    backend: Literal["wan-move-14b", "controlnet", "mock"] = "wan-move-14b"
    resolution: str = "832x480"  # WxH string
    num_frames: int = 81  # 81 frames = ~5 s at 16 fps
    device: str = "cuda"  # "cuda" | "cpu"

    # Model cache
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "motion-mirror"
    )

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_dir_name

    @property
    def resolution_wh(self) -> tuple[int, int]:
        w, h = self.resolution.split("x")
        return int(w), int(h)

    def model_cache(self, sub: str) -> Path:
        p = self.cache_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        return p
