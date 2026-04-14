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
    # "auto"          — detect VRAM at runtime and pick the best option
    # "wan-move-14b"  — 14B I2V, ~22 GB VRAM (sequential CPU offload)
    # "wan-move-fast" — 14B LightX2V 4-step distilled, ~22 GB VRAM, ~5x faster
    # "wan-1.3b-vace" — 1.3B VACE, ~8 GB VRAM (recommended for consumer GPUs)
    # "controlnet"    — deprecated alias for "wan-1.3b-vace"
    # "mock"          — solid-colour video, no GPU required
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

    # VRAM optimisation flags (v0.2a)
    offload_model: bool = False  # sequential layer-by-layer CPU offload (saves VRAM, slower)
    t5_cpu: bool = False         # keep T5 text encoder on CPU (~12 GB VRAM saved)

    # Optional stage upgrades (v0.2a)
    flow_estimator: Literal["farneback", "raft"] = "farneback"
    segmenter: Literal["rembg", "sam2"] = "rembg"

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
