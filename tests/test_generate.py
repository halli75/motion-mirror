"""Tests for generation backends."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.controlnet import generate_with_controlnet
from motion_mirror.generate.models import GenerationRequest
from motion_mirror.generate.wan_move import generate_with_wan_move
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


def _write_traj_npz(path: Path, density: int = 32) -> None:
    np.savez_compressed(
        path,
        density=np.array(density),
        tracks=np.zeros((2, density, 2), dtype=np.float32),
        flow_fields=np.zeros((1, 8, 8, 2), dtype=np.float32),
        frame_w=np.array(8),
        frame_h=np.array(8),
    )


def test_wan_move_returns_generation_result(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert isinstance(result, GenerationResult)


def test_wan_move_output_file_exists(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert result.video_path.exists()


def test_wan_move_output_is_readable_video(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=4)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == req.frames


def test_wan_move_result_metadata(tmp_path):
    req = _mock_request(tmp_path, resolution="128x64", frames=5)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_wan_move(req, cfg)
    assert result.backend == "mock"
    assert result.resolution == "128x64"
    assert result.num_frames == 5


def test_wan_move_creates_output_dir(tmp_path):
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
    result = generate_with_wan_move(req, cfg)
    assert result.video_path.exists()


def test_wan_move_different_seeds_produce_different_colours(tmp_path):
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
    r0 = generate_with_wan_move(req0, cfg)
    r1 = generate_with_wan_move(req1, cfg)

    cap0 = cv2.VideoCapture(str(r0.video_path))
    cap1 = cv2.VideoCapture(str(r1.video_path))
    _, f0 = cap0.read()
    _, f1 = cap1.read()
    cap0.release()
    cap1.release()
    assert not np.allclose(f0.mean(), f1.mean(), atol=1.0)


def test_wan_move_invalid_resolution_raises(tmp_path):
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
        generate_with_wan_move(req, cfg)


def test_wan_move_real_path_requires_weights(tmp_path):
    req = GenerationRequest(
        segmented_image_path=tmp_path / "seg.png",
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="wan-move-14b",
        resolution="128x64",
        frames=2,
        device="cpu",
    )
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="wan-move-14b", device="cpu")
    with pytest.raises((NotImplementedError, FileNotFoundError)):
        generate_with_wan_move(req, cfg)


def test_wan_move_fast_requires_fast_assets(tmp_path):
    seg = tmp_path / "seg.png"
    traj = tmp_path / "traj.npz"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    _write_traj_npz(traj)

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=traj,
        output_path=tmp_path / "fast.mp4",
        backend="wan-move-fast",
        resolution="128x64",
        frames=4,
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-move-fast",
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="wan-move-fast|fast"):
        generate_with_wan_move(req, cfg)


def test_wan_move_fast_requires_lightx2v_runtime(tmp_path):
    seg = tmp_path / "seg.png"
    traj = tmp_path / "traj.npz"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    _write_traj_npz(traj)

    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-move-fast",
        device="cpu",
    )
    model_dir = cfg.model_cache("wan-move-fast")
    for filename in (
        "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "config.json",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1_VAE.pth",
    ):
        (model_dir / filename).write_bytes(b"x")
    (model_dir / "google").mkdir()
    (model_dir / "xlm-roberta-large").mkdir()

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=traj,
        output_path=tmp_path / "fast.mp4",
        backend="wan-move-fast",
        resolution="128x64",
        frames=4,
        device="cpu",
    )

    with patch_sys_modules({"lightx2v": None}):
        with pytest.raises(ImportError, match="LightX2V fast backend"):
            generate_with_wan_move(req, cfg)


def test_wan_move_fast_calls_lightx2v_pipeline(tmp_path):
    seg = tmp_path / "seg.png"
    traj = tmp_path / "traj.npz"
    Image.new("RGBA", (48, 48), (255, 100, 10, 255)).save(seg)
    _write_traj_npz(traj, density=64)

    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-move-fast",
        device="cpu",
        offload_model=True,
        t5_cpu=True,
    )
    model_dir = cfg.model_cache("wan-move-fast")
    for filename in (
        "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "config.json",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1_VAE.pth",
    ):
        (model_dir / filename).write_bytes(b"x")
    (model_dir / "google").mkdir()
    (model_dir / "xlm-roberta-large").mkdir()

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=traj,
        output_path=tmp_path / "fast.mp4",
        backend="wan-move-fast",
        resolution="128x64",
        frames=4,
        device="cpu",
        seed=11,
    )

    class FakeLightX2VPipeline:
        last_instance = None

        def __init__(self, model_path: str, model_cls: str, task: str) -> None:
            self.model_path = model_path
            self.model_cls = model_cls
            self.task = task
            self.offload_kwargs = None
            self.generator_kwargs = None
            self.generate_kwargs = None
            FakeLightX2VPipeline.last_instance = self

        def enable_offload(self, **kwargs):
            self.offload_kwargs = kwargs

        def create_generator(self, **kwargs):
            self.generator_kwargs = kwargs

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            Path(kwargs["save_result_path"]).write_bytes(b"fake-video")

    fake_lightx2v = types.ModuleType("lightx2v")
    fake_lightx2v.LightX2VPipeline = FakeLightX2VPipeline

    with (
        patch_sys_modules({"lightx2v": fake_lightx2v}),
        patch("motion_mirror.generate.wan_move._resolve_lightx2v_attention_backend", return_value="torch"),
    ):
        result = generate_with_wan_move(req, cfg)

    assert result.backend == "wan-move-fast"
    assert result.video_path.exists()
    assert FakeLightX2VPipeline.last_instance is not None
    pipe = FakeLightX2VPipeline.last_instance
    assert pipe.model_path == str(model_dir)
    assert pipe.model_cls == "wan2.1"
    assert pipe.task == "i2v"
    assert pipe.offload_kwargs == {
        "cpu_offload": True,
        "offload_granularity": "block",
        "text_encoder_offload": True,
        "image_encoder_offload": False,
        "vae_offload": False,
    }
    config_json_path = Path(pipe.generator_kwargs["config_json"])
    assert config_json_path.exists()
    runtime_cfg = json.loads(config_json_path.read_text(encoding="utf-8"))
    assert runtime_cfg["infer_steps"] == 4
    assert runtime_cfg["target_video_length"] == 4
    assert runtime_cfg["target_height"] == 64
    assert runtime_cfg["target_width"] == 128
    assert runtime_cfg["enable_cfg"] is False
    assert runtime_cfg["cpu_offload"] is True
    assert runtime_cfg["t5_cpu_offload"] is True
    assert runtime_cfg["self_attn_1_type"] == "torch"
    assert runtime_cfg["cross_attn_1_type"] == "torch"
    assert runtime_cfg["cross_attn_2_type"] == "torch"
    assert pipe.generate_kwargs["seed"] == 11
    assert pipe.generate_kwargs["save_result_path"] == str(req.output_path)


def test_controlnet_returns_generation_result(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    assert isinstance(result, GenerationResult)


def test_controlnet_output_file_exists(tmp_path):
    req = _mock_request(tmp_path)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    assert result.video_path.exists()


def test_controlnet_output_is_readable_video(tmp_path):
    req = _mock_request(tmp_path, frames=3)
    cfg = _mock_cfg(tmp_path)
    result = generate_with_controlnet(req, cfg)
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == req.frames


def test_controlnet_real_path_requires_conditioning_inputs(tmp_path):
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(seg)
    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "out.mp4",
        backend="wan-1.3b-vace",
        resolution="128x64",
        frames=2,
        device="cpu",
    )
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="wan-1.3b-vace", device="cpu")
    with pytest.raises(ValueError, match="conditioning video"):
        generate_with_controlnet(req, cfg)


def test_controlnet_real_path_requires_weights(tmp_path):
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
        frames=4,
        device="cpu",
    )
    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-1.3b-vace",
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="wan-1.3b-vace"):
        generate_with_controlnet(req, cfg)


def test_controlnet_real_path_calls_vace_pipeline(tmp_path):
    seg = tmp_path / "seg.png"
    Image.new("RGBA", (48, 48), (255, 100, 10, 255)).save(seg)
    pose_video = tmp_path / "pose.mp4"
    pose_mask = tmp_path / "mask.mp4"
    _write_rgb_video(pose_video, frames=4, size=(48, 48), value=255)
    _write_rgb_video(pose_mask, frames=4, size=(48, 48), value=0)

    cfg = MotionMirrorConfig(
        project_root=tmp_path,
        cache_dir=tmp_path / "cache",
        backend="wan-1.3b-vace",
        device="cpu",
    )
    model_dir = cfg.model_cache("wan-1.3b-vace")
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    req = GenerationRequest(
        segmented_image_path=seg,
        trajectory_map_path=tmp_path / "traj.npz",
        output_path=tmp_path / "generated.mp4",
        conditioning_video_path=pose_video,
        conditioning_mask_path=pose_mask,
        backend="wan-1.3b-vace",
        resolution="64x64",
        frames=4,
        device="cpu",
        seed=7,
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    fake_torch.Generator = lambda device: FakeGenerator(device)

    class FakeAutoencoderKLWan:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    class FakeScheduler:
        @classmethod
        def from_config(cls, config, flow_shift):
            return types.SimpleNamespace(config=config, flow_shift=flow_shift)

    class FakePipe:
        last_instance = None

        def __init__(self) -> None:
            self.scheduler = types.SimpleNamespace(config={"name": "scheduler"})
            self.vae_scale_factor_spatial = 8
            self.transformer = types.SimpleNamespace(
                config=types.SimpleNamespace(patch_size=(1, 2))
            )
            self.calls = []
            self.device = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            inst = cls()
            cls.last_instance = inst
            inst.pretrained_args = (args, kwargs)
            return inst

        def enable_attention_slicing(self, value):
            self.attention_slicing = value

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            frames = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(kwargs["num_frames"])]
            return types.SimpleNamespace(frames=[frames])

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKLWan = FakeAutoencoderKLWan
    fake_diffusers.WanVACEPipeline = FakePipe
    fake_schedulers_pkg = types.ModuleType("diffusers.schedulers")
    fake_scheduler_module = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    fake_scheduler_module.UniPCMultistepScheduler = FakeScheduler

    with patch_sys_modules(
        {
            "torch": fake_torch,
            "diffusers": fake_diffusers,
            "diffusers.schedulers": fake_schedulers_pkg,
            "diffusers.schedulers.scheduling_unipc_multistep": fake_scheduler_module,
        }
    ):
        result = generate_with_controlnet(req, cfg)

    assert result.backend == "wan-1.3b-vace"
    assert result.video_path.exists()
    assert FakePipe.last_instance is not None
    call = FakePipe.last_instance.calls[0]
    assert len(call["video"]) == 4
    assert len(call["mask"]) == 4
    assert len(call["reference_images"]) == 1
    assert call["num_frames"] == 4


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
