from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

BackendName = Literal[
    "auto",
    "wan-1.3b-vace",
    "wan-14b-vace",
    "wan-14b-vace-gguf",
    "mock",
]

DeviceName = Literal["cuda", "cpu"]
FlowEstimatorName = Literal["farneback", "raft"]
SegmenterName = Literal["rembg", "sam2"]


@dataclass(slots=True)
class MotionMirrorConfig:
    project_root: Path = field(default_factory=Path.cwd)
    output_dir_name: str = "outputs"

    # Trajectory
    trajectory_density: int = 512  # 512 = default, 1024 = HQ

    # Generation backend
    # "auto"               - detect VRAM at runtime and pick the best option
    # "wan-1.3b-vace"      - 1.3B VACE motion transfer, needs ~9 GB free VRAM
    # "wan-14b-vace"       - full Wan2.1-VACE-14B-diffusers (~75 GB download).
    #                        Explicit-only via --backend; NOT selected by auto.
    #                        GPU-unvalidated as of 0.4.0a0; community-estimated
    #                        ~8-12 GB peak VRAM at 480p with offload_model+t5_cpu.
    # "wan-14b-vace-gguf"  - QuantStack Q4_K_M quantized 14B transformer
    #                        (~11.6 GB download vs ~75 GB full) + base components.
    #                        Explicit-only via --backend; NOT selected by auto.
    #                        GPU-unvalidated as of 0.4.0a0; community-estimated
    #                        ~8-12 GB peak VRAM at 480p with offload_model+t5_cpu.
    # "mock"               - solid-colour video, no GPU required
    backend: BackendName = "wan-1.3b-vace"

    resolution: str = "832x480"  # WxH string
    num_frames: int = 81         # 81 frames = ~5 s at 16 fps
    # Denoising steps; higher = sharper, slower. None = resolved default:
    # 30 normally, or the per-backend fast-mode step count when fast=True.
    num_inference_steps: int | None = None
    # Classifier-free guidance scale. None = backend default (5.0). Distilled
    # few-step generation wants 1.0 (CFG off).
    guidance_scale: float | None = None
    # Optional LoRA applied to the transformer before generation. Accepts a
    # local .safetensors path, a HF repo id, or "repo_id:filename".
    # Not supported on GGUF-quantized backends (diffusers limitation).
    lora: str | None = None
    lora_scale: float = 1.0
    # Curated fast/distilled generation: per-backend distill artifact,
    # few-step defaults, CFG off. Mutually exclusive with `lora`.
    fast: bool = False
    device: str = "cuda"         # "cuda" | "cpu"

    # VRAM optimization flags
    offload_model: bool = False  # sequential layer-by-layer CPU offload
    t5_cpu: bool = False         # keep T5 text encoder on CPU when supported

    # Optional stage upgrades
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
        if self.num_inference_steps is not None and not 1 <= self.num_inference_steps <= 200:
            raise ValueError(
                f"num_inference_steps must be in [1, 200], got "
                f"{self.num_inference_steps}"
            )
        if self.guidance_scale is not None and self.guidance_scale <= 0:
            raise ValueError(
                f"guidance_scale must be > 0, got {self.guidance_scale}"
            )
        if self.lora_scale <= 0:
            raise ValueError(
                f"lora_scale must be > 0, got {self.lora_scale}"
            )
        if self.fast and self.lora is not None:
            raise ValueError(
                "fast and lora are mutually exclusive: fast applies a curated "
                "distill LoRA; set one or the other"
            )

        for field_name, allowed in (
            ("backend", get_args(BackendName)),
            ("device", get_args(DeviceName)),
            ("flow_estimator", get_args(FlowEstimatorName)),
            ("segmenter", get_args(SegmenterName)),
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(
                    f"Invalid {field_name} {value!r}. "
                    f"Allowed values: {list(allowed)}."
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
