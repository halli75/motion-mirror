"""Tests for the VACE generation backend."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.generate.models import GenerationRequest
from motion_mirror.generate.vace import generate_with_vace
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


# ── mock backend ────────────────────────────────────────────────────────────


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


# ── real (VACE) path ─────────────────────────────────────────────────────────


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


# ── fake diffusers pipeline harness ──────────────────────────────────────────


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class FakeAutoencoderKLWan:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return object()


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

    def to(self, device):
        self.device = device
        return self

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
    fake_schedulers_pkg = types.ModuleType("diffusers.schedulers")
    fake_scheduler_module = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    fake_scheduler_module.UniPCMultistepScheduler = FakeScheduler
    return {
        "torch": _fake_torch(cuda_available),
        "diffusers": fake_diffusers,
        "diffusers.schedulers": fake_schedulers_pkg,
        "diffusers.schedulers.scheduling_unipc_multistep": fake_scheduler_module,
    }


def _vace_pipeline_request(tmp_path: Path, **cfg_overrides) -> tuple[GenerationRequest, MotionMirrorConfig]:
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
        backend="wan-1.3b-vace",
        **cfg_overrides,
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
