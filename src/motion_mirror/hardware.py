"""GPU detection and backend recommendation for Motion Mirror v0.2a.

This module is deliberately import-safe: it never imports torch at module
level so the rest of the package can load without a GPU or CUDA installation.
All GPU queries happen inside functions and are guarded by try/except.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from .exceptions import InsufficientVRAMError


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
    """Return a :class:`GPUInfo` for device 0, or ``None`` if no CUDA GPU.

    Never raises — any exception is caught and ``None`` is returned so that
    callers without a GPU (CI, mock runs) work without special-casing.
    """
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


# VRAM thresholds (GB) for each backend.
# These are conservative estimates with sequential CPU offloading active.
_BACKEND_VRAM: dict[str, float] = {
    "wan-move-14b":   22.0,   # 14B bfloat16, sequential offload
    "wan-move-fast":  22.0,   # LightX2V 4-step, same weights size
    "wan-1.3b-vace":   8.0,   # 1.3B VACE
}
_MIN_VRAM_GB = 8.0  # absolute floor — below this, nothing will work


def recommend_backend(vram_gb: float) -> tuple[str, dict]:
    """Return ``(backend_name, config_overrides)`` for *vram_gb* of free VRAM.

    Parameters
    ----------
    vram_gb:
        Free VRAM in GB as reported by ``get_gpu_info().free_vram_gb``.

    Returns
    -------
    tuple[str, dict]
        Backend name and a dict of :class:`~motion_mirror.config.MotionMirrorConfig`
        keyword overrides to apply (e.g. ``{"offload_model": True}``).

    Raises
    ------
    InsufficientVRAMError
        If *vram_gb* is below the absolute minimum of 8 GB.

    Examples
    --------
    >>> recommend_backend(32.0)
    ('wan-move-14b', {})
    >>> recommend_backend(12.0)
    ('wan-1.3b-vace', {})
    >>> recommend_backend(9.0)
    ('wan-1.3b-vace', {'offload_model': True})
    """
    if vram_gb >= 22.0:
        return "wan-move-14b", {}
    if vram_gb >= 12.0:
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
    """Resolve ``backend='auto'`` to a concrete backend based on VRAM.

    Parameters
    ----------
    base:
        Config with ``backend == 'auto'``.  All other fields are preserved.

    Returns
    -------
    MotionMirrorConfig
        A new config with ``backend`` resolved.  If no GPU is detected the
        original config is returned unchanged with a warning.

    Raises
    ------
    InsufficientVRAMError
        If a GPU is detected but has less than 8 GB free VRAM.
    """
    # Import here to avoid circular dependency (config imports nothing from hardware)
    from .config import MotionMirrorConfig

    if base.backend != "auto":
        return base

    info = get_gpu_info()
    if info is None:
        warnings.warn(
            "backend='auto' requested but no CUDA GPU detected. "
            "Falling back to 'wan-move-14b'. Use --backend mock for CPU-only testing.",
            UserWarning,
            stacklevel=3,
        )
        return MotionMirrorConfig(
            **{f: getattr(base, f) for f in base.__slots__ if f != "backend"},
            backend="wan-move-14b",
        )

    backend, overrides = recommend_backend(info.free_vram_gb)

    # Build a new config with the resolved values
    kwargs = {f: getattr(base, f) for f in base.__slots__}
    kwargs["backend"] = backend
    kwargs.update(overrides)
    return MotionMirrorConfig(**kwargs)
