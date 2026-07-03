"""Tests for GPU detection and backend recommendation."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.exceptions import HardwareError, InsufficientVRAMError, MotionMirrorError
from motion_mirror.hardware import (
    _BACKEND_TIERS,
    GPUInfo,
    auto_config,
    get_gpu_info,
    recommend_backend,
)


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
    backend, overrides = recommend_backend(40.0)
    assert backend == "wan-move-14b"
    assert overrides == {}


def test_recommend_backend_just_below_full_model_threshold():
    backend, overrides = recommend_backend(39.9)
    assert backend == "wan-move-fast"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_24gb_fast():
    backend, overrides = recommend_backend(24.0)
    assert backend == "wan-move-fast"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_just_below_fast_threshold():
    backend, overrides = recommend_backend(23.9)
    assert backend == "wan-move-gguf"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_16gb_gguf():
    backend, overrides = recommend_backend(16.0)
    assert backend == "wan-move-gguf"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_just_below_gguf_threshold():
    backend, overrides = recommend_backend(12.2)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"t5_cpu": True}


def test_recommend_backend_13gb_routes_to_gguf():
    backend, overrides = recommend_backend(13.0)
    assert backend == "wan-move-gguf"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_11_5_falls_to_min_vace():
    backend, overrides = recommend_backend(11.5)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_12_1_mid_vace_t5_cpu_only():
    backend, overrides = recommend_backend(12.1)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"t5_cpu": True}


def test_recommend_backend_12gb_vace_t5_cpu():
    backend, overrides = recommend_backend(12.0)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"t5_cpu": True}


def test_recommend_backend_just_below_mid_threshold():
    backend, overrides = recommend_backend(11.9)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_9_5gb_activates_offload():
    backend, overrides = recommend_backend(9.5)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_9_02gb_boundary():
    backend, overrides = recommend_backend(9.02)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_insufficient_raises():
    with pytest.raises(InsufficientVRAMError) as exc_info:
        recommend_backend(8.5)
    assert exc_info.value.available_gb == pytest.approx(8.5)
    assert exc_info.value.required_gb == pytest.approx(9.02)
    message = str(exc_info.value)
    assert "9.02 GB" in message
    assert "wan-1.3b-vace" in message


def test_backend_tiers_strictly_descend_by_floor():
    floors = [tier.minimum_vram_gb for tier in _BACKEND_TIERS]
    assert floors == sorted(floors, reverse=True)
    assert len(set(floors)) == len(floors)


def test_recommend_backend_zero_vram_raises():
    with pytest.raises(InsufficientVRAMError):
        recommend_backend(0.0)


def test_auto_config_without_gpu_raises():
    cfg = MotionMirrorConfig(backend="auto")
    with patch("motion_mirror.hardware.get_gpu_info", return_value=None):
        with pytest.raises(InsufficientVRAMError) as exc_info:
            auto_config(cfg)
    assert exc_info.value.available_gb == 0.0
    assert exc_info.value.required_gb == pytest.approx(9.02)
    message = str(exc_info.value)
    assert "no CUDA GPU" in message
    assert "--backend mock" in message


@pytest.mark.parametrize(
    ("free_vram_gb", "expected_backend", "expected_offload", "expected_t5_cpu"),
    [
        (40.0, "wan-move-14b", False, False),
        (39.9, "wan-move-fast", True, True),
        (24.0, "wan-move-fast", True, True),
        (23.9, "wan-move-gguf", True, True),
        (16.0, "wan-move-gguf", True, True),
        (13.0, "wan-move-gguf", True, True),
        (12.2, "wan-1.3b-vace", False, True),
        (12.0, "wan-1.3b-vace", False, True),
        (11.9, "wan-1.3b-vace", True, True),
        (9.5, "wan-1.3b-vace", True, True),
    ],
)
def test_auto_config_resolves_mocked_vram_boundaries(
    free_vram_gb,
    expected_backend,
    expected_offload,
    expected_t5_cpu,
):
    cfg = MotionMirrorConfig(backend="auto", offload_model=False, t5_cpu=False)
    gpu = GPUInfo(
        name="Mock GPU",
        total_vram_gb=max(40.0, free_vram_gb),
        free_vram_gb=free_vram_gb,
    )
    with patch("motion_mirror.hardware.get_gpu_info", return_value=gpu):
        resolved = auto_config(cfg)
    assert resolved.backend == expected_backend
    assert resolved.offload_model is expected_offload
    assert resolved.t5_cpu is expected_t5_cpu


def test_recommend_backend_24gb():
    backend, overrides = recommend_backend(24.0)
    assert backend == "wan-move-fast"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_just_below_threshold():
    backend, overrides = recommend_backend(23.9)
    assert backend == "wan-move-gguf"
    assert overrides == {"offload_model": True, "t5_cpu": True}


def test_recommend_backend_12gb():
    backend, overrides = recommend_backend(12.0)
    assert backend == "wan-1.3b-vace"
    assert overrides == {"t5_cpu": True}


def test_auto_config_resolves_backend_and_overrides():
    cfg = MotionMirrorConfig(backend="auto", offload_model=False, t5_cpu=False)
    gpu = GPUInfo(name="RTX 4060", total_vram_gb=8.0, free_vram_gb=9.5)
    with patch("motion_mirror.hardware.get_gpu_info", return_value=gpu):
        resolved = auto_config(cfg)
    assert resolved.backend == "wan-1.3b-vace"
    assert resolved.offload_model is True
    assert resolved.t5_cpu is True


def test_insufficient_vram_error_is_hardware_error():
    exc = InsufficientVRAMError("test", available_gb=2.0, required_gb=8.0)
    assert isinstance(exc, HardwareError)
    assert isinstance(exc, MotionMirrorError)
