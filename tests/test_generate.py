"""Tests for the VACE generation backend."""
from __future__ import annotations

import re
import sys
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.models import GenerationRequest
from motion_mirror.generate.vace import generate_with_vace, resolve_generation_settings
from motion_mirror.types import GenerationResult


def _mock_request(tmp_path: Path, resolution: str = "128x64", frames: int = 4, seed: int = 0) -> GenerationRequest:
    return GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated.mp4",
        backend="mock",
        resolution=resolution,
        frames=frames,
        device="cpu",
        seed=seed,
    )


def _mock_cfg(tmp_path: Path) -> MotionMirrorConfig:
    return MotionMirrorConfig(project_root=tmp_path, backend="mock", device="cpu")


def _write_rgb_video(path: Path, frames: int = 4, size: tuple[int, int] = (64, 64), value: int = 255) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, size)
    for _ in range(frames):
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_vace_mock_returns_generation_result(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_vace(req, cfg)
    assert isinstance(result, GenerationResult)


def test_vace_mock_output_file_exists(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_vace(req, cfg)
    assert result.video_path.exists()


def test_vace_mock_output_is_readable_video(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=4)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_vace(req, cfg)
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == req.frames


def test_vace_mock_result_metadata(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=5)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_vace(req, cfg)
    assert result.backend == "mock"
    assert result.resolution == "128x64"
    assert result.num_frames == 5


def test_vace_mock_creates_output_dir(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "deep" / "nested" / "generated.mp4",
        backend="mock",
        resolution="64x64",
        frames=2,
        device="cpu",
    )
    cfg = _mock_cfg(tmp_path)
    result = generate_with_vace(req, cfg)
    assert result.video_path.exists()


def test_vace_mock_different_seeds_produce_different_colours(tmp_path):
    req0 = _mock_request(tmp_path, seed=0)
    req1 = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated_1.mp4",
        backend="mock",
        resolution="128x64",
        frames=4,
        device="cpu",
        seed=99,
    )
    cfg = _mock_cfg(tmp_path)
    r0 = generate_with_vace(req0, cfg)
    r1 = generate_with_vace(req1, cfg)

    cap0 = cv2.VideoCapture(str(r0.video_path))
    cap1 = cv2.VideoCapture(str(r1.video_path))
    _, f0 = cap0.read()
    _, f1 = cap1.read()
    cap0.release()
    cap1.release()
    assert not np.allclose(f0.mean(), f1.mean(), atol=1.0)


def test_vace_mock_invalid_resolution_raises(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="mock",
        resolution="bad_resolution",
        frames=2,
        device="cpu",
    )
    cfg = _mock_cfg(tmp_path)
    with pytest.raises(ValueError, match="Invalid resolution"):
        generate_with_vace(req, cfg)


def test_vace_real_path_requires_conditioning_inputs(tmp_path):
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="wan-1.3b-vace",
        resolution="128x64",
        frames=5,
        device="cpu",
    )
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="wan-1.3b-vace", device="cpu")
    with pytest.raises(ValueError, match="conditioning video"):
        generate_with_vace(req, cfg)


def test_vace_real_path_requires_weights(tmp_path):
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video)
    _write_rgb_video(pose_mask, value=0)

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend="wan-1.3b-vace",
        resolution="128x64",
        frames=5,
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-1.3b-vace",
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="wan-1.3b-vace"):
        generate_with_vace(req, cfg)


def test_vace_rejects_non_4k_plus_1_frames(tmp_path):
    """The real VACE path must reject frame counts that aren't 4k+1."""
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video)
    _write_rgb_video(pose_mask, value=0)

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend="wan-1.3b-vace",
        resolution="128x64",
        frames=80,  # 80 - 1 = 79, not divisible by 4
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-1.3b-vace",
        device="cpu",
    )
    with pytest.raises(ValueError, match="4k\\+1"):
        generate_with_vace(req, cfg)


def test_vace_accepts_4k_plus_1_frames_and_proceeds(tmp_path):
    """frames=81 passes the frame check and only then fails on missing weights."""
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video)
    _write_rgb_video(pose_mask, value=0)

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend="wan-1.3b-vace",
        resolution="128x64",
        frames=81,  # 81 - 1 = 80, divisible by 4
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-1.3b-vace",
        device="cpu",
    )
    # Gets past the 4k+1 frame check, then trips on the missing weights.
    with pytest.raises(FileNotFoundError, match="wan-1.3b-vace"):
        generate_with_vace(req, cfg)


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class FakeAutoencoderKLWan:
    last_kwargs: dict | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_kwargs = kwargs
        return object()


class FakeGGUFQuantizationConfig:
    last_instance = None

    def __init__(self, compute_dtype=None):
        self.compute_dtype = compute_dtype
        FakeGGUFQuantizationConfig.last_instance = self


class FakeWanVACETransformer3DModel:
    last_args: tuple | None = None
    last_kwargs: dict | None = None

    @classmethod
    def from_single_file(cls, *args, **kwargs):
        cls.last_args = args
        cls.last_kwargs = kwargs
        return types.SimpleNamespace(_sentinel="fake-gguf-transformer")


class FakeScheduler:
    @classmethod
    def from_config(cls, config, flow_shift):
        return types.SimpleNamespace(config=config, flow_shift=flow_shift)


class FakeTextEncoder:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class FakePipe:
    last_instance = None

    def __init__(self) -> None:
        self.scheduler = types.SimpleNamespace(config={"name": "scheduler"})
        self.vae_scale_factor_spatial = 8
        self.transformer = types.SimpleNamespace(
            config=types.SimpleNamespace(patch_size=(1, 2))
        )
        self.text_encoder = FakeTextEncoder()
        self.calls = []
        self.device = None
        self.attention_slicing = None
        self.sequential_offload_called = False
        self.model_cpu_offload_called = False
        # Ordered log of lifecycle calls, for asserting e.g. that LoRA
        # fusing happens before any device placement / offload.
        self.events = []
        FakePipe.last_instance = self

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inst = cls()
        cls.last_instance = inst
        inst.pretrained_args = (args, kwargs)
        return inst

    def enable_attention_slicing(self, value):
        self.attention_slicing = value

    def enable_sequential_cpu_offload(self):
        self.sequential_offload_called = True
        self.events.append(("enable_sequential_cpu_offload",))

    def enable_model_cpu_offload(self):
        self.model_cpu_offload_called = True
        self.events.append(("enable_model_cpu_offload",))

    def to(self, device):
        self.device = device
        self.events.append(("to", device))
        return self

    def load_lora_weights(self, path, **kwargs):
        self.events.append(("load_lora_weights", str(path)))

    def fuse_lora(self, lora_scale=1.0):
        self.events.append(("fuse_lora", lora_scale))

    def unload_lora_weights(self):
        self.events.append(("unload_lora_weights",))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(kwargs["num_frames"])]
        return types.SimpleNamespace(frames=[frames])


def _fake_torch(cuda_available: bool):
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available, empty_cache=lambda: None
    )
    fake_torch.Generator = lambda device: FakeGenerator(device)
    return fake_torch


def _fake_diffusers_modules(cuda_available: bool) -> dict[str, object]:
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan = FakeAutoencoderKLWan
    fake_diffusers.WanVACEPipeline = FakePipe
    fake_diffusers.GGUFQuantizationConfig = FakeGGUFQuantizationConfig
    fake_diffusers.WanVACETransformer3DModel = FakeWanVACETransformer3DModel
    fake_schedulers_pkg = types.ModuleType("diffusers.schedulers")
    fake_scheduler_module = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    fake_scheduler_module.UniPCMultistepScheduler = FakeScheduler
    return {
        "torch": _fake_torch(cuda_available),
        "diffusers": fake_diffusers,
        "diffusers.schedulers": fake_schedulers_pkg,
        "diffusers.schedulers.scheduling_unipc_multistep": fake_scheduler_module,
        # The gguf backend import-checks the gguf package up front.
        "gguf": types.ModuleType("gguf"),
    }


def _seed_model_index(cfg: MotionMirrorConfig, subdir: str) -> Path:
    """Write a minimal diffusers model_index.json into a cache subdir."""
    model_dir = cfg.model_cache(subdir)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    return model_dir


def _seed_gguf_transformer(cfg: MotionMirrorConfig, spec_subdir: str, filename: str) -> Path:
    """Write a non-empty fake .gguf transformer file into a cache subdir."""
    gguf_dir = cfg.model_cache(spec_subdir)
    path = gguf_dir / filename
    path.write_bytes(b"\x00" * 32)
    return path


def _vace_pipeline_request(
    tmp_path: Path, *, backend: str = "wan-1.3b-vace", **cfg_overrides
) -> tuple[GenerationRequest, MotionMirrorConfig]:
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (48, 48), (255, 100, 10, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video, frames=5, size=(48, 48), value=255)
    _write_rgb_video(pose_mask, frames=5, size=(48, 48), value=0)

    cfg_overrides.setdefault("device", "cpu")
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend=backend,
        **cfg_overrides,
    )

    # Seed whatever weights the chosen backend expects to find on disk.
    if backend == "wan-14b-vace-gguf":
        _seed_gguf_transformer(cfg, "wan-14b-vace-gguf", "Wan2.1_14B_VACE-Q4_K_M.gguf")
        # Provide the base components via a complete full-14B cache.
        _seed_model_index(cfg, "wan-14b-vace")
    elif backend == "wan-14b-vace":
        _seed_model_index(cfg, "wan-14b-vace")
    else:
        _seed_model_index(cfg, "wan-1.3b-vace")

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend=backend,
        resolution="64x64",
        frames=5,
        device=cfg_overrides["device"],
        seed=7,
    )
    return req, cfg


def test_vace_real_path_calls_vace_pipeline(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        result = generate_with_vace(req, cfg)

    assert result.backend == "wan-1.3b-vace"
    assert result.video_path.exists()
    assert FakePipe.last_instance is not None
    call = FakePipe.last_instance.calls[0]
    assert len(call["video"]) == 5
    assert len(call["mask"]) == 5
    assert len(call["reference_images"]) == 1
    assert call["num_frames"] == 5
    # Default steps flow through to the pipe call.
    assert call["num_inference_steps"] == 30


def test_vace_pipe_receives_configured_inference_steps(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, num_inference_steps=42)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    call = FakePipe.last_instance.calls[0]
    assert call["num_inference_steps"] == 42


def test_vace_pipe_guidance_scale_defaults_to_five(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    call = FakePipe.last_instance.calls[0]
    assert call["guidance_scale"] == 5.0


def test_vace_pipe_receives_configured_guidance_scale(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, guidance_scale=1.0)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    call = FakePipe.last_instance.calls[0]
    assert call["guidance_scale"] == 1.0


def _seed_fast_14b_lora(cfg: MotionMirrorConfig) -> Path:
    path = (
        cfg.model_cache("wan-fast-14b-lora")
        / "loras"
        / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 16)
    return path


def test_resolve_generation_settings_normal_defaults():
    cfg = MotionMirrorConfig(device="cpu")
    assert resolve_generation_settings(cfg, "wan-1.3b-vace") == (30, 5.0, None)


def test_resolve_generation_settings_fast_14b_defaults():
    cfg = MotionMirrorConfig(device="cpu", fast=True, backend="wan-14b-vace")
    assert resolve_generation_settings(cfg, "wan-14b-vace") == (4, 1.0, 5.0)


def test_resolve_generation_settings_explicit_values_beat_fast():
    cfg = MotionMirrorConfig(
        device="cpu", fast=True, num_inference_steps=20, guidance_scale=3.0
    )
    assert resolve_generation_settings(cfg, "wan-14b-vace") == (20, 3.0, 5.0)


def test_resolve_generation_settings_fast_1_3b_defaults():
    cfg = MotionMirrorConfig(device="cpu", fast=True)
    assert resolve_generation_settings(cfg, "wan-1.3b-vace") == (4, 1.0, 5.0)


def test_resolve_generation_settings_fast_gguf_uses_eight_steps():
    cfg = MotionMirrorConfig(device="cpu", fast=True, backend="wan-14b-vace-gguf")
    assert resolve_generation_settings(cfg, "wan-14b-vace-gguf") == (8, 1.0, 5.0)


def test_resolve_generation_settings_fast_unsupported_backend_raises():
    cfg = MotionMirrorConfig(device="cpu", fast=True)
    with pytest.raises(ValueError, match="not supported"):
        resolve_generation_settings(cfg, "wan-99b-vace")


def test_resolve_generation_settings_fast_mock_is_noop():
    cfg = MotionMirrorConfig(device="cpu", fast=True, backend="mock")
    assert resolve_generation_settings(cfg, "mock") == (30, 5.0, None)


def test_vace_fast_14b_applies_lora_steps_guidance_shift(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace", fast=True)
    lora_path = _seed_fast_14b_lora(cfg)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    call = pipe.calls[0]
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0
    assert pipe.scheduler.flow_shift == 5.0
    assert pipe.events[:3] == [
        ("load_lora_weights", str(lora_path)),
        ("fuse_lora", 1.0),
        ("unload_lora_weights",),
    ]


def test_vace_fast_14b_missing_lora_names_download_command(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace", fast=True)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.raises(
            FileNotFoundError, match="download --model wan-fast-14b-lora"
        ):
            generate_with_vace(req, cfg)


def _seed_fast_1_3b_lora(cfg: MotionMirrorConfig) -> Path:
    path = (
        cfg.model_cache("wan-fast-1.3b")
        / "LoRAs"
        / "Wan2_1_self_forcing_1_3B"
        / "Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 16)
    return path


def test_vace_fast_1_3b_applies_lora_and_warns_noncommercial(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, fast=True)
    lora_path = _seed_fast_1_3b_lora(cfg)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.warns(UserWarning, match="NON-COMMERCIAL"):
            generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    call = pipe.calls[0]
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0
    assert pipe.scheduler.flow_shift == 5.0
    assert pipe.events[0] == ("load_lora_weights", str(lora_path))


def test_vace_fast_1_3b_missing_lora_names_artifact_and_nc(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, fast=True)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.raises(FileNotFoundError) as exc_info:
            generate_with_vace(req, cfg)

    msg = str(exc_info.value)
    assert "download --model wan-fast-1.3b" in msg
    assert "NON-COMMERCIAL" in msg


def test_vace_fast_gguf_swaps_in_fusionx_transformer(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace-gguf", fast=True
    )
    fusionx_path = _seed_gguf_transformer(
        cfg, "wan-14b-vace-fusionx-gguf", "Wan2.1_T2V_14B_FusionX_VACE-Q4_K_M.gguf"
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.warns(UserWarning, match="EXPERIMENTAL"):
            generate_with_vace(req, cfg)

    assert FakeWanVACETransformer3DModel.last_args[0] == str(fusionx_path)
    call = FakePipe.last_instance.calls[0]
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 1.0


def test_vace_fast_gguf_missing_fusionx_names_download_command(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace-gguf", fast=True
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.raises(
            FileNotFoundError, match="wan-14b-vace-fusionx-gguf"
        ):
            generate_with_vace(req, cfg)


def test_vace_normal_gguf_path_unchanged_by_fast_feature(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace-gguf")

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    assert "Wan2.1_14B_VACE-Q4_K_M.gguf" in FakeWanVACETransformer3DModel.last_args[0]
    call = FakePipe.last_instance.calls[0]
    assert call["num_inference_steps"] == 30
    assert call["guidance_scale"] == 5.0


def test_vace_fast_14b_emits_no_license_warning(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace", fast=True)
    _seed_fast_14b_lora(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
            generate_with_vace(req, cfg)

    assert not any("NON-COMMERCIAL" in str(w.message) for w in caught)


def _seed_lora_file(tmp_path: Path) -> Path:
    lora_file = tmp_path / "my_lora.safetensors"
    lora_file.write_bytes(b"\x00" * 16)
    return lora_file


def test_vace_no_lora_makes_no_lora_calls(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path)

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    events = FakePipe.last_instance.events
    assert not any(e[0].endswith("lora_weights") or e[0] == "fuse_lora" for e in events)


def test_vace_local_lora_loads_fuses_unloads_in_order(tmp_path):
    lora_file = _seed_lora_file(tmp_path)
    req, cfg = _vace_pipeline_request(
        tmp_path, lora=str(lora_file), lora_scale=0.8
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    events = FakePipe.last_instance.events
    assert events[:3] == [
        ("load_lora_weights", str(lora_file)),
        ("fuse_lora", 0.8),
        ("unload_lora_weights",),
    ]


def test_vace_lora_applied_before_offload(tmp_path):
    """LoRA fuse must precede sequential offload (meta-tensor hazard)."""
    lora_file = _seed_lora_file(tmp_path)
    req, cfg = _vace_pipeline_request(
        tmp_path, device="cuda", offload_model=True, lora=str(lora_file)
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    events = FakePipe.last_instance.events
    names = [e[0] for e in events]
    assert names.index("fuse_lora") < names.index("enable_sequential_cpu_offload")


def test_vace_lora_on_gguf_backend_raises(tmp_path):
    lora_file = _seed_lora_file(tmp_path)
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace-gguf", lora=str(lora_file)
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.raises(ValueError, match="GGUF"):
            generate_with_vace(req, cfg)


def test_vace_missing_local_lora_file_raises(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, lora=str(tmp_path / "nope.safetensors")
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        with pytest.raises(FileNotFoundError, match="LoRA"):
            generate_with_vace(req, cfg)


def test_vace_offload_model_enables_sequential_cpu_offload(tmp_path):
    """offload_model=True on CUDA must route through sequential CPU offload."""
    req, cfg = _vace_pipeline_request(
        tmp_path, device="cuda", offload_model=True, t5_cpu=True
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    assert pipe is not None
    assert pipe.sequential_offload_called is True
    # Sequential offload owns device placement, so no manual pipe.to / t5 move.
    assert pipe.device is None


def test_vace_t5_cpu_moves_text_encoder_without_offload(tmp_path):
    """t5_cpu=True (offload off) must keep the text encoder on CPU."""
    req, cfg = _vace_pipeline_request(
        tmp_path, device="cuda", offload_model=False, t5_cpu=True
    )

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    assert pipe is not None
    assert pipe.sequential_offload_called is False
    assert pipe.device == "cuda"
    assert pipe.text_encoder.device == "cpu"


def test_vace_empties_cuda_cache_after_success(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, device="cuda")
    modules = _fake_diffusers_modules(cuda_available=True)
    empty_cache_calls = []
    modules["torch"].cuda.empty_cache = lambda: empty_cache_calls.append(1)

    with patch_sys_modules(modules):
        generate_with_vace(req, cfg)

    # One pre-load clear + one post-generation cleanup.
    assert len(empty_cache_calls) == 2


def test_vace_empties_cuda_cache_when_generation_raises(tmp_path):
    """VRAM must be released even when the pipeline call itself fails."""
    req, cfg = _vace_pipeline_request(tmp_path, device="cuda")
    modules = _fake_diffusers_modules(cuda_available=True)
    empty_cache_calls = []
    modules["torch"].cuda.empty_cache = lambda: empty_cache_calls.append(1)

    class ExplodingPipe(FakePipe):
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    modules["diffusers"].WanVACEPipeline = ExplodingPipe

    with patch_sys_modules(modules):
        with pytest.raises(RuntimeError, match="out of memory"):
            generate_with_vace(req, cfg)

    # The pre-load clear plus the finally-block cleanup — without the
    # try/finally only the pre-load clear would run.
    assert len(empty_cache_calls) == 2


class patch_sys_modules:
    def __init__(self, modules: dict[str, object]) -> None:
        self.modules = modules
        self.previous: dict[str, object] = {}

    def __enter__(self):
        for name, module in self.modules.items():
            self.previous[name] = sys.modules.get(name)  # type: ignore[assignment]
            sys.modules[name] = module  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]
        for name in self.modules:
            if name not in self.previous:
                sys.modules.pop(name, None)
        return False


def _bare_vace_request(tmp_path: Path, backend: str, frames: int = 5) -> tuple[GenerationRequest, MotionMirrorConfig]:
    """A real-path request with conditioning inputs but NO seeded weights."""
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video)
    _write_rgb_video(pose_mask, value=0)

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend=backend,
        resolution="128x64",
        frames=frames,
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend=backend,
        device="cpu",
    )
    return req, cfg


def test_vace_14b_dispatch_uses_14b_cache_and_backend(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace")

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        result = generate_with_vace(req, cfg)

    assert result.backend == "wan-14b-vace"
    assert result.video_path.exists()
    # WanVACEPipeline.from_pretrained was pointed at the 14B cache dir.
    args, _kwargs = FakePipe.last_instance.pretrained_args
    assert "wan-14b-vace" in str(args[0])
    assert "wan-1.3b-vace" not in str(args[0])


@pytest.mark.parametrize(
    "backend,group",
    [
        ("wan-1.3b-vace", "vace"),
        ("wan-14b-vace", "vace-14b"),
        ("wan-14b-vace-gguf", "vace-14b-gguf"),
    ],
)
def test_vace_missing_weights_hints_download_group(tmp_path, backend, group):
    req, cfg = _bare_vace_request(tmp_path, backend)
    with pytest.raises(FileNotFoundError, match=re.escape(f"--model {group}")):
        generate_with_vace(req, cfg)


def test_vace_gguf_loader_uses_exact_args(tmp_path):
    # CUDA path: the production config — dtype follows the device (bf16 on
    # cuda, fp32 on cpu), so assert against the cuda run the pod will do.
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace-gguf", device="cuda")

    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        result = generate_with_vace(req, cfg)

    assert result.backend == "wan-14b-vace-gguf"
    kw = FakeWanVACETransformer3DModel.last_kwargs
    assert kw is not None
    assert kw["config"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert kw["subfolder"] == "transformer"
    assert kw["torch_dtype"] == "bfloat16"
    # GGUF quantization config built with bf16 compute dtype and passed through.
    assert kw["quantization_config"] is FakeGGUFQuantizationConfig.last_instance
    assert FakeGGUFQuantizationConfig.last_instance.compute_dtype == "bfloat16"
    # The transformer path (positional arg 0) points at the .gguf file.
    assert "Wan2.1_14B_VACE-Q4_K_M.gguf" in str(FakeWanVACETransformer3DModel.last_args[0])
    # The loaded transformer is injected into the pipeline.
    _args, pkwargs = FakePipe.last_instance.pretrained_args
    assert "transformer" in pkwargs
    assert getattr(pkwargs["transformer"], "_sentinel", None) == "fake-gguf-transformer"


def test_vace_gguf_missing_transformer_file_raises(tmp_path):
    req, cfg = _bare_vace_request(tmp_path, "wan-14b-vace-gguf")
    # Seed only the base components; leave the .gguf transformer absent.
    _seed_model_index(cfg, "wan-14b-vace")
    with pytest.raises(FileNotFoundError, match="Wan2.1_14B_VACE-Q4_K_M.gguf"):
        generate_with_vace(req, cfg)


def _gguf_cfg(tmp_path: Path) -> MotionMirrorConfig:
    return MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-14b-vace-gguf",
        device="cpu",
    )


def test_resolve_gguf_base_source_full_cache(tmp_path):
    from motion_mirror.generate import vace

    cfg = _gguf_cfg(tmp_path)
    spec = vace._VACE_BACKEND_SPECS["wan-14b-vace-gguf"]
    full_dir = _seed_model_index(cfg, "wan-14b-vace")
    assert vace._resolve_gguf_base_source(cfg, spec) == str(full_dir)


def test_resolve_gguf_base_source_base_complete(tmp_path):
    from motion_mirror.generate import vace

    cfg = _gguf_cfg(tmp_path)
    spec = vace._VACE_BACKEND_SPECS["wan-14b-vace-gguf"]
    base_dir = _seed_model_index(cfg, "wan-14b-vace-base")
    assert vace._resolve_gguf_base_source(cfg, spec) == str(base_dir)


def test_resolve_gguf_base_source_both_absent_returns_hub_id(tmp_path, capsys):
    from motion_mirror.generate import vace

    cfg = _gguf_cfg(tmp_path)
    spec = vace._VACE_BACKEND_SPECS["wan-14b-vace-gguf"]
    result = vace._resolve_gguf_base_source(cfg, spec)
    assert result == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    # An info line warns about the impending multi-GB download.
    assert "multi-GB" in capsys.readouterr().out


def test_resolve_gguf_base_source_base_partial_raises(tmp_path):
    from motion_mirror.generate import vace

    cfg = _gguf_cfg(tmp_path)
    spec = vace._VACE_BACKEND_SPECS["wan-14b-vace-gguf"]
    base_dir = cfg.model_cache("wan-14b-vace-base")
    (base_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match=re.escape("--model vace-14b-gguf")):
        vace._resolve_gguf_base_source(cfg, spec)


def test_memory_policy_14b_full_offload_uses_sequential(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace", device="cuda", offload_model=True, t5_cpu=True
    )
    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    assert pipe.sequential_offload_called is True
    assert pipe.model_cpu_offload_called is False


def test_memory_policy_gguf_offload_uses_whole_module(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace-gguf", device="cuda", offload_model=True, t5_cpu=True
    )
    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    # GGUF must use whole-module offload and NEVER sequential (quant metadata).
    assert pipe.model_cpu_offload_called is True
    assert pipe.sequential_offload_called is False


def test_memory_policy_gguf_no_offload_honors_to_device_and_t5(tmp_path):
    req, cfg = _vace_pipeline_request(
        tmp_path, backend="wan-14b-vace-gguf", device="cuda", offload_model=False, t5_cpu=True
    )
    with patch_sys_modules(_fake_diffusers_modules(cuda_available=True)):
        generate_with_vace(req, cfg)

    pipe = FakePipe.last_instance
    assert pipe.model_cpu_offload_called is False
    assert pipe.sequential_offload_called is False
    assert pipe.device == "cuda"
    assert pipe.text_encoder.device == "cpu"


@pytest.mark.parametrize("backend", ["wan-1.3b-vace", "wan-14b-vace", "wan-14b-vace-gguf"])
def test_vace_uses_fp32_vae(tmp_path, backend):
    req, cfg = _vace_pipeline_request(tmp_path, backend=backend)
    with patch_sys_modules(_fake_diffusers_modules(cuda_available=False)):
        generate_with_vace(req, cfg)

    assert FakeAutoencoderKLWan.last_kwargs is not None
    assert FakeAutoencoderKLWan.last_kwargs["subfolder"] == "vae"
    assert FakeAutoencoderKLWan.last_kwargs["torch_dtype"] == "float32"


@pytest.mark.parametrize("backend", ["wan-14b-vace", "wan-14b-vace-gguf"])
def test_vace_new_backends_reject_non_4k_plus_1_frames(tmp_path, backend):
    req, cfg = _bare_vace_request(tmp_path, backend, frames=80)
    with pytest.raises(ValueError, match="4k\\+1"):
        generate_with_vace(req, cfg)


def test_vace_gguf_empties_cuda_cache_when_generation_raises(tmp_path):
    req, cfg = _vace_pipeline_request(tmp_path, backend="wan-14b-vace-gguf", device="cuda")
    modules = _fake_diffusers_modules(cuda_available=True)
    empty_cache_calls = []
    modules["torch"].cuda.empty_cache = lambda: empty_cache_calls.append(1)

    class ExplodingPipe(FakePipe):
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    modules["diffusers"].WanVACEPipeline = ExplodingPipe

    with patch_sys_modules(modules):
        with pytest.raises(RuntimeError, match="out of memory"):
            generate_with_vace(req, cfg)

    # Pre-load clear + finally-block cleanup (which also releases the transformer).
    assert len(empty_cache_calls) == 2


def test_vace_prompt_never_names_the_control_modality():
    # The text prompt dominates subject choice: naming the conditioning signal
    # ("skeleton") made VACE render an anatomical skeleton instead of the
    # reference character (Wave-2 GPU run, 2026-07-03). The prompt must
    # describe the desired subject, never the mechanism.
    from motion_mirror.generate import vace

    banned = ("skeleton", "pose", "bones", "control", "stick figure")
    prompt = vace._VACE_PROMPT.lower()
    for word in banned:
        assert word not in prompt, f"control-modality word {word!r} in VACE prompt"

    assert "reference image" in prompt

    negative = vace._NEGATIVE_PROMPT.lower()
    for word in ("skeleton", "x-ray", "stick figure"):
        assert word in negative, f"{word!r} missing from negative prompt"
