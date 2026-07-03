from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.concat_id import generate_with_concat_id
from motion_mirror.generate.models import GenerationRequest


def _request(tmp_path: Path) -> GenerationRequest:
    image_path = tmp_path / "identity.png"
    Image.new("RGB", (32, 32), (200, 100, 80)).save(image_path)
    traj_path = tmp_path / "trajectory.npz"
    np.savez(traj_path, density=16)
    return GenerationRequest(
        segmented_image_path=image_path,
        identity_image_path=image_path,
        trajectory_map_path=traj_path,
        output_path=tmp_path / "generated.mp4",
        backend="wan-1.3b-concat-id",
        resolution="64x32",
        frames=3,
        device="cpu",
        seed=123,
    )


def _cfg(tmp_path: Path) -> MotionMirrorConfig:
    return MotionMirrorConfig(
        backend="wan-1.3b-concat-id",
        resolution="64x32",
        num_frames=3,
        trajectory_density=16,
        device="cpu",
        cache_dir=tmp_path / "cache",
        project_root=tmp_path,
    )


def _write_concat_id_assets(cfg: MotionMirrorConfig) -> Path:
    model_dir = cfg.model_cache("wan-1.3b-concat-id")
    (model_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"base")
    (model_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"t5")
    (model_dir / "Wan2.1_VAE.pth").write_bytes(b"vae")
    (model_dir / "second_stage_adaln.pt").write_bytes(b"adapter")
    (model_dir / "google").mkdir()
    (model_dir / "google" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "antelopev2").mkdir()
    (model_dir / "antelopev2" / "model.onnx").write_bytes(b"onnx")
    return model_dir


def _install_fake_runtime(monkeypatch, pipeline_call):
    cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache_calls=0)

    def _empty_cache():
        cuda.empty_cache_calls += 1

    cuda.empty_cache = _empty_cache
    fake_torch = types.SimpleNamespace(bfloat16="bfloat16", float32="float32", cuda=cuda)

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls()

        def load_concat_id(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            return pipeline_call(**kwargs)

    diffsynth_mod = types.ModuleType("diffsynth")
    pipelines_mod = types.ModuleType("diffsynth.pipelines")
    wan_video_mod = types.ModuleType("diffsynth.pipelines.wan_video")
    wan_video_mod.ModelConfig = FakeModelConfig
    wan_video_mod.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffsynth", diffsynth_mod)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", pipelines_mod)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", wan_video_mod)
    return cuda


def test_concat_id_missing_weights_error_is_clear(tmp_path):
    req = _request(tmp_path)
    cfg = _cfg(tmp_path)

    with pytest.raises(FileNotFoundError, match="motion-mirror download --model concat-id"):
        generate_with_concat_id(req, cfg)


def test_concat_id_missing_dependency_error_is_lazy(tmp_path, monkeypatch):
    req = _request(tmp_path)
    cfg = _cfg(tmp_path)
    _write_concat_id_assets(cfg)
    monkeypatch.setitem(sys.modules, "diffsynth", None)

    with pytest.raises(ImportError, match="DiffSynth-Studio"):
        generate_with_concat_id(req, cfg)


def test_concat_id_fake_runtime_call_writes_video(tmp_path, monkeypatch):
    req = _request(tmp_path)
    cfg = _cfg(tmp_path)
    assets_dir = _write_concat_id_assets(cfg)
    calls: dict[str, object] = {}

    class FakeTorch:
        bfloat16 = "bfloat16"
        float32 = "float32"
        cuda = types.SimpleNamespace(is_available=lambda: False)

    class FakeModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWanVideoPipeline:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["from_pretrained"] = kwargs
            return cls()

        def load_concat_id(self, **kwargs):
            calls["load_concat_id"] = kwargs

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return [
                Image.new("RGB", (64, 32), (index * 40, 20, 200))
                for index in range(3)
            ]

    diffsynth_mod = types.ModuleType("diffsynth")
    pipelines_mod = types.ModuleType("diffsynth.pipelines")
    wan_video_mod = types.ModuleType("diffsynth.pipelines.wan_video")
    wan_video_mod.ModelConfig = FakeModelConfig
    wan_video_mod.WanVideoPipeline = FakeWanVideoPipeline
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "diffsynth", diffsynth_mod)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", pipelines_mod)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video", wan_video_mod)

    result = generate_with_concat_id(req, cfg)

    assert result.backend == "wan-1.3b-concat-id"
    assert result.video_path.exists()
    assert calls["load_concat_id"] == {
        "adapter_path": str(assets_dir / "second_stage_adaln.pt"),
        "face_models_dir": str(assets_dir / "antelopev2"),
    }
    generate_call = calls["generate"]
    assert generate_call["height"] == 32
    assert generate_call["width"] == 64
    assert generate_call["num_frames"] == 3
    assert generate_call["seed"] == 123
    assert generate_call["face_image"].size == (32, 32)


def test_concat_id_empties_cuda_cache_after_success(tmp_path, monkeypatch):
    req = _request(tmp_path)
    cfg = _cfg(tmp_path)
    _write_concat_id_assets(cfg)
    cuda = _install_fake_runtime(
        monkeypatch,
        lambda **kwargs: [Image.new("RGB", (64, 32), (10, 20, 200)) for _ in range(3)],
    )

    generate_with_concat_id(req, cfg)

    assert cuda.empty_cache_calls == 1


def test_concat_id_empties_cuda_cache_when_generation_raises(tmp_path, monkeypatch):
    req = _request(tmp_path)
    cfg = _cfg(tmp_path)
    _write_concat_id_assets(cfg)

    def _boom(**kwargs):
        raise RuntimeError("CUDA out of memory during Concat-ID generation")

    cuda = _install_fake_runtime(monkeypatch, _boom)

    with pytest.raises(RuntimeError, match="CUDA out of memory during Concat-ID generation"):
        generate_with_concat_id(req, cfg)

    assert cuda.empty_cache_calls == 1
