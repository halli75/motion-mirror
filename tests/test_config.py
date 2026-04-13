import tempfile
from pathlib import Path

import pytest

from motion_mirror.config import MotionMirrorConfig


def test_defaults():
    c = MotionMirrorConfig()
    assert c.trajectory_density == 512
    assert c.backend == "wan-move-14b"
    assert c.resolution == "832x480"
    assert c.num_frames == 81
    assert c.device == "cuda"
    assert c.output_dir_name == "outputs"


def test_output_dir():
    c = MotionMirrorConfig(project_root=Path("/tmp/proj"))
    assert c.output_dir == Path("/tmp/proj/outputs")


def test_resolution_wh():
    c = MotionMirrorConfig(resolution="832x480")
    assert c.resolution_wh == (832, 480)

    c2 = MotionMirrorConfig(resolution="1280x720")
    assert c2.resolution_wh == (1280, 720)


def test_resolution_wh_invalid():
    c = MotionMirrorConfig(resolution="bad")
    with pytest.raises(ValueError):
        _ = c.resolution_wh


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
        p1 = c.model_cache("wan-move")
        p2 = c.model_cache("wan-move")
        assert p1 == p2
        assert p1.exists()


def test_mock_backend():
    c = MotionMirrorConfig(backend="mock", device="cpu")
    assert c.backend == "mock"
    assert c.device == "cpu"
