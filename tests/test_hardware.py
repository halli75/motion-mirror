"""Tests for the hardware detection and backend recommendation module (Phase A)."""
from __future__ import annotations

import pytest

from motion_mirror.exceptions import InsufficientVRAMError
from motion_mirror.hardware import GPUInfo, get_gpu_info, recommend_backend


# ── GPUInfo dataclass ─────────────────────────────────────────────────────────

def test_gpu_info_used_vram():
    info = GPUInfo(name="Test GPU", total_vram_gb=24.0, free_vram_gb=20.0)
    assert abs(info.used_vram_gb - 4.0) < 1e-6


# ── get_gpu_info — no GPU environment ─────────────────────────────────────────

def test_get_gpu_info_returns_none_or_gpuinfo():
    """In CI (no GPU) returns None; on a GPU machine returns a GPUInfo."""
    result = get_gpu_info()
    assert result is None or isinstance(result, GPUInfo)


def test_get_gpu_info_never_raises():
    """Must not raise regardless of environment."""
    try:
        get_gpu_info()
    except Exception as exc:
        pytest.fail(f"get_gpu_info() raised unexpectedly: {exc}")


# ── recommend_backend ─────────────────────────────────────────────────────────

def test_recommend_backend_high_vram():
    backend, overrides = recommend_backend(32.0)
    assert backend == "wan-move-14b"
    assert overrides == {}


def test_recommend_backend_24gb():
    backend, overrides = recommend_backend(24.0)
    assert backend == "wan-move-14b"
    assert overrides == {}


def test_recommend_backend_just_below_threshold():
    """21 GB falls below the 22 GB threshold for 14B — should pick 1.3B."""
    backend, overrides = recommend_backend(21.9)
    assert backend == "wan-1.3b-vace"


def test_recommend_backend_12gb():
    backend, overrides = recommend_backend(12.0)
    assert backend == "wan-1.3b-vace"
    assert overrides == {}


def test_recommend_backend_9gb_activates_offload():
    backend, overrides = recommend_backend(9.0)
    assert backend == "wan-1.3b-vace"
    assert overrides.get("offload_model") is True


def test_recommend_backend_8gb_boundary():
    backend, overrides = recommend_backend(8.0)
    assert backend == "wan-1.3b-vace"
    assert overrides.get("offload_model") is True


def test_recommend_backend_insufficient_raises():
    with pytest.raises(InsufficientVRAMError) as exc_info:
        recommend_backend(4.0)
    assert exc_info.value.available_gb == pytest.approx(4.0)
    assert exc_info.value.required_gb == 8.0


def test_recommend_backend_zero_vram_raises():
    with pytest.raises(InsufficientVRAMError):
        recommend_backend(0.0)


def test_insufficient_vram_error_is_hardware_error():
    from motion_mirror.exceptions import HardwareError, MotionMirrorError
    exc = InsufficientVRAMError("test", available_gb=2.0, required_gb=8.0)
    assert isinstance(exc, HardwareError)
    assert isinstance(exc, MotionMirrorError)
