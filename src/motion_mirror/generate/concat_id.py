"""Experimental Concat-ID Wan 1.3B identity backend."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest

_BACKEND = "wan-1.3b-concat-id"
_BASE_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
_ADAPTER_MODEL_ID = "yongzhong/Concat-ID-Wan"
_DEFAULT_PROMPT = (
    "A person with the same identity as the reference image, natural motion, "
    "stable face, clean anatomy, cinematic lighting."
)


@dataclass(slots=True)
class ConcatIDAssets:
    model_dir: Path
    adapter_path: Path
    face_models_dir: Path


def generate_with_concat_id(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate with the experimental Concat-ID Wan 1.3B backend.

    Concat-ID's public Wan release targets Wan2.1-T2V-1.3B through DiffSynth
    runtime code. It is intentionally separate from the VACE backend until a
    real compatibility path is proven.
    """
    cfg = config or MotionMirrorConfig(backend=_BACKEND)
    out_w, out_h = _parse_resolution(request.resolution)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_inputs(request)
    assets = _resolve_assets(cfg)
    runtime = _load_runtime()

    identity_image_path = request.identity_image_path or request.segmented_image_path
    identity_image = runtime["Image"].open(identity_image_path).convert("RGB")
    pipe = _build_pipeline(runtime, cfg, assets)
    _load_identity_adapter(pipe, assets)

    output = pipe(
        prompt=_DEFAULT_PROMPT,
        face_image=identity_image,
        height=out_h,
        width=out_w,
        num_frames=request.frames,
        num_inference_steps=50,
        cfg_scale=5.0,
        seed=request.seed,
    )
    _write_output_video(request.output_path, output)

    return GenerationResult(
        video_path=request.output_path,
        backend=_BACKEND,
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        w_str, h_str = resolution.split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {resolution!r}. Expected 'WxH', e.g. '832x480'."
        ) from exc


def _validate_inputs(request: GenerationRequest) -> None:
    for label, path in (
        ("segmented image", request.segmented_image_path),
        ("trajectory map", request.trajectory_map_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Concat-ID input {label} not found: {path}")
    if request.identity_image_path is not None and not request.identity_image_path.exists():
        raise FileNotFoundError(
            f"Concat-ID identity image not found: {request.identity_image_path}"
        )


def _resolve_assets(config: MotionMirrorConfig) -> ConcatIDAssets:
    model_dir = config.model_cache(_BACKEND)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Concat-ID weights not found in {model_dir}.\n"
            "Run: motion-mirror download --model concat-id"
        )

    adapter_path = _first_existing(
        model_dir,
        ("second_stage_adaln.pt", "first_stage.pt"),
    )
    if adapter_path is None:
        raise FileNotFoundError(
            f"Concat-ID adapter weights are missing in {model_dir}. "
            "Expected second_stage_adaln.pt or first_stage.pt."
        )

    required = (
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "google",
    )
    missing = [rel for rel in required if not (model_dir / rel).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Concat-ID base Wan2.1-T2V-1.3B assets are incomplete in {model_dir}: {joined}."
        )

    face_models_dir = model_dir / "antelopev2"
    if not face_models_dir.exists():
        raise FileNotFoundError(
            f"Concat-ID face model directory is missing: {face_models_dir}"
        )

    return ConcatIDAssets(
        model_dir=model_dir,
        adapter_path=adapter_path,
        face_models_dir=face_models_dir,
    )


def _load_runtime() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
        from diffsynth.pipelines.wan_video import (  # type: ignore[import]
            ModelConfig,
            WanVideoPipeline,
        )
    except ImportError as exc:
        raise ImportError(
            "Concat-ID requires torch, Pillow, and DiffSynth-Studio.\n"
            "Install optional dependencies with: pip install -e '.[concat-id]'"
        ) from exc
    return {
        "torch": torch,
        "Image": Image,
        "ModelConfig": ModelConfig,
        "WanVideoPipeline": WanVideoPipeline,
    }


def _build_pipeline(
    runtime: dict[str, Any],
    config: MotionMirrorConfig,
    assets: ConcatIDAssets,
) -> object:
    torch = runtime["torch"]
    model_config = runtime["ModelConfig"]
    pipeline_cls = runtime["WanVideoPipeline"]
    device = config.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if not hasattr(pipeline_cls, "from_pretrained"):
        raise RuntimeError(
            "Installed DiffSynth WanVideoPipeline does not expose from_pretrained()."
        )

    return pipeline_cls.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            model_config(
                model_id=str(assets.model_dir),
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            model_config(
                model_id=str(assets.model_dir),
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            model_config(
                model_id=str(assets.model_dir),
                origin_file_pattern="Wan2.1_VAE.pth",
            ),
        ],
        tokenizer_config=model_config(
            model_id=str(assets.model_dir),
            origin_file_pattern="google/**",
        ),
    )


def _load_identity_adapter(pipe: object, assets: ConcatIDAssets) -> None:
    if hasattr(pipe, "load_concat_id"):
        pipe.load_concat_id(
            adapter_path=str(assets.adapter_path),
            face_models_dir=str(assets.face_models_dir),
        )
        return
    raise RuntimeError(
        "Installed Concat-ID/DiffSynth runtime does not expose load_concat_id(). "
        "Motion Mirror keeps this backend experimental until the runtime API is stable."
    )


def _write_output_video(output_path: Path, output: object) -> None:
    if output_path.exists():
        return
    frames = getattr(output, "frames", output)
    frames_list = frames if isinstance(frames, list) else list(frames)
    if not frames_list:
        raise RuntimeError("Concat-ID returned no frames")

    first = np.array(frames_list[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        16.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

    try:
        for frame in frames_list:
            arr = np.array(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _first_existing(base_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    return None
