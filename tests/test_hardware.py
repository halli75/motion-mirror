"""Tests for GPU detection and backend recommendation."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.exceptions import HardwareError, InsufficientVRAMError, MotionMirrorError
from motion_mirror.hardware import GPUInfo, auto_config, get_gpu_info, recommend_backend


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


def test_recommend_backend_high_vram():
    backend, overrides = recommend_backend(32.0)
    assert backend == "wan-move-14b"
    assert overrides == {}


def test_recommend_backend_24gb():
    backend, overrides = recommend_backend(24.0)
    assert backend == "wan-move-14b"
    assert overrides == {}


def test_recommend_backend_just_below_threshold():
    backend, overrides = recommend_backend(23.9)
    assert backend == "wan-1.3b-vace"
    assert overrides == {}


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


def test_auto_config_without_gpu_raises():
    cfg = MotionMirrorConfig(backend="auto")
    with patch("motion_mirror.hardware.get_gpu_info", return_value=None):
        with pytest.raises(InsufficientVRAMError) as exc_info:
            auto_config(cfg)
    assert exc_info.value.available_gb == 0.0
    assert exc_info.value.required_gb == 8.0


def test_auto_config_resolves_backend_and_overrides():
    cfg = MotionMirrorConfig(backend="auto", offload_model=False, t5_cpu=False)
    gpu = GPUInfo(name="RTX 4060", total_vram_gb=8.0, free_vram_gb=9.5)
    with patch("motion_mirror.hardware.get_gpu_info", return_value=gpu):
        resolved = auto_config(cfg)
    assert resolved.backend == "wan-1.3b-vace"
    assert resolved.offload_model is True
    assert resolved.t5_cpu is False


def test_insufficient_vram_error_is_hardware_error():
    exc = InsufficientVRAMError("test", available_gb=2.0, required_gb=8.0)
    assert isinstance(exc, HardwareError)
    assert isinstance(exc, MotionMirrorError)
