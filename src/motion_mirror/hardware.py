"""GPU detection and backend recommendation for Motion Mirror v0.2a."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .exceptions import InsufficientVRAMError

# Safety margin over measured peak VRAM.
_HEADROOM_GB = 1.0

_FULL_MODEL_VRAM_GB = 40.0
_FAST_MODEL_VRAM_GB = 24.0
# Peaks measured in v0.2a GPU validation (RTX 4090/3090, 17-frame smoke,
# offload_model + t5_cpu): gguf 11.28 GB, vace 8.02 GB.
# A 12 GB t5_cpu-only middle tier was probed on GPU 2026-07-03 and removed:
# pipe.to(cuda) transiently peaked at 15.07 GB (T5 lands on GPU before the
# t5_cpu move) and generation then crashed at VAE encode (bf16 input into the
# fp32 VAE without accelerate's auto-casting offload hooks). 12 GB cards use
# the fully-offloaded vace tier below.
_GGUF_MODEL_VRAM_GB = 11.28 + _HEADROOM_GB
_MIN_VRAM_GB = 8.02 + _HEADROOM_GB


@dataclass(frozen=True)
class BackendTier:
    """A VRAM floor and concrete backend selected by backend='auto'."""

    minimum_vram_gb: float
    backend: str
    overrides: tuple[tuple[str, bool], ...] = ()


_BACKEND_TIERS: tuple[BackendTier, ...] = (
    BackendTier(_FULL_MODEL_VRAM_GB, "wan-move-14b"),
    BackendTier(_FAST_MODEL_VRAM_GB, "wan-move-fast", (("offload_model", True), ("t5_cpu", True))),
    BackendTier(_GGUF_MODEL_VRAM_GB, "wan-move-gguf", (("offload_model", True), ("t5_cpu", True))),
    BackendTier(_MIN_VRAM_GB, "wan-1.3b-vace", (("offload_model", True), ("t5_cpu", True))),
)


@dataclass
class GPUInfo:
    """Snapshot of GPU hardware state."""

    name: str
    total_vram_gb: float
    free_vram_gb: float

    @property
    def used_vram_gb(self) -> float:
        return self.total_vram_gb - self.free_vram_gb


def get_gpu_info() -> GPUInfo | None:
    """Return a GPUInfo for device 0, or None if no CUDA GPU is available."""
    try:
        import torch  # type: ignore[import]

        if not torch.cuda.is_available():
            return None

        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, _total = torch.cuda.mem_get_info(idx)
        return GPUInfo(
            name=props.name,
            total_vram_gb=props.total_memory / 1e9,
            free_vram_gb=free_bytes / 1e9,
        )
    except Exception:
        return None


def recommend_backend(vram_gb: float) -> tuple[str, dict]:
    """Return (backend_name, config_overrides) for the available free VRAM."""
    for tier in _BACKEND_TIERS:
        if vram_gb >= tier.minimum_vram_gb:
            return tier.backend, dict(tier.overrides)

    raise InsufficientVRAMError(
        f"Only {vram_gb:.1f} GB VRAM free. "
        f"Motion Mirror auto-selection requires at least {_MIN_VRAM_GB:.2f} GB "
        "free VRAM for the lightest backend (wan-1.3b-vace). "
        "Free VRAM by closing other applications and try again.",
        available_gb=vram_gb,
        required_gb=_MIN_VRAM_GB,
    )


def auto_config(base: "MotionMirrorConfig") -> "MotionMirrorConfig":  # noqa: F821
    """Resolve backend='auto' to a concrete backend based on available VRAM."""
    from .config import MotionMirrorConfig

    if base.backend != "auto":
        return base

    info = get_gpu_info()
    if info is None:
        raise InsufficientVRAMError(
            "backend='auto' requested but no CUDA GPU was detected. "
            f"Auto-selection requires a CUDA GPU with at least {_MIN_VRAM_GB:.2f} GB free VRAM. "
            "Use --backend mock for CPU-only testing.",
            available_gb=0.0,
            required_gb=_MIN_VRAM_GB,
        )

    backend, overrides = recommend_backend(info.free_vram_gb)
    return dataclasses.replace(base, backend=backend, **overrides)
