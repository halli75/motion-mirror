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

    # Generation backend
    # "auto"          - detect VRAM at runtime and pick the best option
    # "wan-move-14b"  - 14B I2V, ~24 GB VRAM
    # "wan-move-fast" - true LightX2V 4-step Wan2.1 I2V fast backend
    # "wan-1.3b-vace" - 1.3B VACE, ~8 GB VRAM
    # "controlnet"    - deprecated alias for "wan-1.3b-vace"
    # "mock"          - solid-colour video, no GPU required
    backend: Literal[
        "auto",
        "wan-move-14b",
        "wan-move-fast",
        "wan-1.3b-vace",
        "controlnet",
        "mock",
    ] = "wan-move-14b"

    resolution: str = "832x480"  # WxH string
    num_frames: int = 81         # 81 frames = ~5 s at 16 fps
    device: str = "cuda"         # "cuda" | "cpu"

    # VRAM optimization flags (v0.2a)
    offload_model: bool = False  # sequential layer-by-layer CPU offload
    t5_cpu: bool = False         # keep T5 text encoder on CPU when supported

    # Optional stage upgrades (v0.2a)
    flow_estimator: Literal["farneback", "raft"] = "farneback"
    segmenter: Literal["rembg", "sam2"] = "rembg"

    # Model cache
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "motion-mirror"
    )

    def __post_init__(self) -> None:
        """Validate configuration values at construction time."""
        try:
            w_str, h_str = self.resolution.split("x")
            w, h = int(w_str), int(h_str)
            if w < 1 or h < 1:
                raise ValueError("dimensions must be positive")
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Invalid resolution {self.resolution!r}. "
                "Expected 'WxH' format with positive integers, e.g. '832x480'."
            ) from exc

        if self.trajectory_density < 1:
            raise ValueError(
                f"trajectory_density must be >= 1, got {self.trajectory_density}"
            )
        if self.num_frames < 1:
            raise ValueError(
                f"num_frames must be >= 1, got {self.num_frames}"
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
