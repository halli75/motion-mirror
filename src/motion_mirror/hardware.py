"""GPU detection and backend recommendation for Motion Mirror v0.2a."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .exceptions import InsufficientVRAMError

_HIGH_END_VRAM_GB = 24.0
_MID_TIER_VRAM_GB = 12.0
_MIN_VRAM_GB = 8.0


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
    if vram_gb >= _HIGH_END_VRAM_GB:
        return "wan-move-14b", {}
    if vram_gb >= _MID_TIER_VRAM_GB:
        return "wan-1.3b-vace", {}
    if vram_gb >= _MIN_VRAM_GB:
        return "wan-1.3b-vace", {"offload_model": True}

    raise InsufficientVRAMError(
        f"Only {vram_gb:.1f} GB VRAM free. "
        f"Motion Mirror requires at least {_MIN_VRAM_GB:.0f} GB for the "
        "lightest backend (wan-1.3b-vace). "
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
            "Use --backend mock for CPU-only testing or run on a CUDA GPU "
            "with at least 8 GB VRAM.",
            available_gb=0.0,
            required_gb=_MIN_VRAM_GB,
        )

    backend, overrides = recommend_backend(info.free_vram_gb)
    return dataclasses.replace(base, backend=backend, **overrides)
