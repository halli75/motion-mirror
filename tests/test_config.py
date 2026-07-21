import tempfile
from pathlib import Path

import pytest

from motion_mirror.config import MotionMirrorConfig


def test_defaults():
    c = MotionMirrorConfig()
    assert c.trajectory_density == 512
    assert c.backend == "wan-1.3b-vace"
    assert c.resolution == "832x480"
    assert c.num_frames == 81
    assert c.num_inference_steps is None  # None = resolved default (30 / fast)
    assert c.device == "cuda"
    assert c.output_dir_name == "outputs"
    assert c.fast is False


def test_num_inference_steps_validation():
    assert MotionMirrorConfig(num_inference_steps=50).num_inference_steps == 50
    with pytest.raises(ValueError):
        MotionMirrorConfig(num_inference_steps=0)
    with pytest.raises(ValueError):
        MotionMirrorConfig(num_inference_steps=201)


def test_guidance_scale_validation():
    assert MotionMirrorConfig().guidance_scale is None
    assert MotionMirrorConfig(guidance_scale=1.0).guidance_scale == 1.0
    with pytest.raises(ValueError, match="guidance_scale"):
        MotionMirrorConfig(guidance_scale=0)
    with pytest.raises(ValueError, match="guidance_scale"):
        MotionMirrorConfig(guidance_scale=-2.5)


def test_lora_defaults_and_validation():
    c = MotionMirrorConfig()
    assert c.lora is None
    assert c.lora_scale == 1.0
    assert MotionMirrorConfig(lora_scale=0.5).lora_scale == 0.5
    with pytest.raises(ValueError, match="lora_scale"):
        MotionMirrorConfig(lora_scale=0)


def test_output_dir():
    c = MotionMirrorConfig(project_root=Path("/tmp/proj"))
    assert c.output_dir == Path("/tmp/proj/outputs")


def test_resolution_wh():
    c = MotionMirrorConfig(resolution="832x480")
    assert c.resolution_wh == (832, 480)

    c2 = MotionMirrorConfig(resolution="1280x720")
    assert c2.resolution_wh == (1280, 720)


def test_resolution_wh_invalid():
    with pytest.raises(ValueError):
        MotionMirrorConfig(resolution="bad")


def test_model_cache_creates_dir():
    with tempfile.TemporaryDirectory() as tmp:
        c = MotionMirrorConfig(cache_dir=Path(tmp))
        p = c.model_cache("dwpose")
        assert p.exists()
        assert p.is_dir()
        assert p == Path(tmp) / "dwpose"


def test_model_cache_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        c = MotionMirrorConfig(cache_dir=Path(tmp))
        p1 = c.model_cache("wan-1.3b-vace")
        p2 = c.model_cache("wan-1.3b-vace")
        assert p1 == p2
        assert p1.exists()


def test_mock_backend():
    c = MotionMirrorConfig(backend="mock", device="cpu")
    assert c.backend == "mock"
    assert c.device == "cpu"


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        MotionMirrorConfig(backend="wan-999")


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="device"):
        MotionMirrorConfig(device="gpu")


def test_invalid_segmenter_raises():
    with pytest.raises(ValueError, match="segmenter"):
        MotionMirrorConfig(segmenter="magic")


def test_invalid_flow_estimator_raises():
    with pytest.raises(ValueError, match="flow_estimator"):
        MotionMirrorConfig(flow_estimator="lucas-kanade")


def test_all_valid_backends_construct():
    for backend in (
        "auto",
        "wan-1.3b-vace",
        "wan-14b-vace",
        "wan-14b-vace-gguf",
        "mock",
    ):
        assert MotionMirrorConfig(backend=backend).backend == backend


def test_valid_optional_stage_values_construct():
    c = MotionMirrorConfig(
        device="cpu",
        flow_estimator="raft",
        segmenter="sam2",
    )
    assert c.flow_estimator == "raft"
    assert c.segmenter == "sam2"
