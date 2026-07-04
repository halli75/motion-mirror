"""Tests for GPU detection and single-tier backend recommendation.

v0.3 is VACE-only: every GPU at or above the floor resolves to the same
fully-offloaded ``wan-1.3b-vace`` config; anything below raises.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.exceptions import HardwareError, InsufficientVRAMError, MotionMirrorError
from motion_mirror.hardware import (
    GPUInfo,
    auto_config,
    get_gpu_info,
    recommend_backend,
)

_FLOOR_GB = 9.02  # 8.02 measured peak + 1.0 headroom
_OVERRIDES = {"offload_model": True, "t5_cpu": True}


def test_gpu_info_used_vram():
    info = GPUInfo(name="Test GPU", total_vram_gb=24.0, free_vram_gb=20.0)
    assert abs(info.used_vram_gb - 4.0) < 1e-6


def test_get_gpu_info_returns_none_or_gpuinfo():
    result = get_gpu_info()
    assert result is None or isinstance(result, GPUInfo)


def test_get_gpu_info_never_raises():
    try:
        get_gpu_info()
    except Exception as exc:
        pytest.fail(f"get_gpu_info() raised unexpectedly: {exc}")


@pytest.mark.parametrize("free_vram_gb", [9.02, 10.0, 24.0, 48.0])
def test_recommend_backend_at_or_above_floor_is_always_vace(free_vram_gb):
    # Single tier: even a 48 GB card returns vace with both memory overrides.
    backend, overrides = recommend_backend(free_vram_gb)
    assert backend == "wan-1.3b-vace"
    assert overrides == _OVERRIDES


def test_recommend_backend_just_below_floor_raises():
    with pytest.raises(InsufficientVRAMError) as exc_info:
        recommend_backend(9.01)
    assert exc_info.value.available_gb == pytest.approx(9.01)
    assert exc_info.value.required_gb == pytest.approx(_FLOOR_GB)
    message = str(exc_info.value)
    assert "9.02 GB" in message
    assert "wan-1.3b-vace" in message


def test_recommend_backend_zero_vram_raises():
    with pytest.raises(InsufficientVRAMError):
        recommend_backend(0.0)


def test_auto_config_without_gpu_raises():
    cfg = MotionMirrorConfig(backend="auto")
    with patch("motion_mirror.hardware.get_gpu_info", return_value=None):
        with pytest.raises(InsufficientVRAMError) as exc_info:
            auto_config(cfg)
    assert exc_info.value.available_gb == 0.0
    assert exc_info.value.required_gb == pytest.approx(_FLOOR_GB)
    message = str(exc_info.value)
    assert "no CUDA GPU" in message
    assert "--backend mock" in message


def test_auto_config_resolves_backend_and_overrides():
    cfg = MotionMirrorConfig(backend="auto", offload_model=False, t5_cpu=False)
    gpu = GPUInfo(name="RTX 4060", total_vram_gb=12.0, free_vram_gb=9.5)
    with patch("motion_mirror.hardware.get_gpu_info", return_value=gpu):
        resolved = auto_config(cfg)
    assert resolved.backend == "wan-1.3b-vace"
    assert resolved.offload_model is True
    assert resolved.t5_cpu is True


def test_auto_config_huge_vram_still_resolves_to_vace_with_overrides():
    # Single-tier semantics: a 48 GB card doesn't unlock a heavier backend.
    cfg = MotionMirrorConfig(backend="auto", offload_model=False, t5_cpu=False)
    gpu = GPUInfo(name="A6000", total_vram_gb=48.0, free_vram_gb=48.0)
    with patch("motion_mirror.hardware.get_gpu_info", return_value=gpu):
        resolved = auto_config(cfg)
    assert resolved.backend == "wan-1.3b-vace"
    assert resolved.offload_model is True
    assert resolved.t5_cpu is True


def test_auto_config_passes_non_auto_backend_through_unchanged():
    cfg = MotionMirrorConfig(backend="mock", device="cpu")
    # get_gpu_info must not even be consulted for a concrete backend.
    with patch("motion_mirror.hardware.get_gpu_info", side_effect=AssertionError("should not probe GPU")):
        resolved = auto_config(cfg)
    assert resolved is cfg
    assert resolved.backend == "mock"


def test_insufficient_vram_error_is_hardware_error():
    exc = InsufficientVRAMError("test", available_gb=2.0, required_gb=8.0)
    assert isinstance(exc, HardwareError)
    assert isinstance(exc, MotionMirrorError)
