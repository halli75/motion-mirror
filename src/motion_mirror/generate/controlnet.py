"""Wan2.1 VACE generation backend."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest

_WAN_VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def generate_with_controlnet(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate via the lightweight Wan VACE backend."""
    cfg = config or MotionMirrorConfig()

    try:
        w_str, h_str = request.resolution.split("x")
        out_w, out_h = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {request.resolution!r}. Expected 'WxH'."
        ) from exc

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "mock" or request.backend == "mock":
        return _generate_mock(request, out_w, out_h)

    return _generate_vace_1b(request, cfg, out_w, out_h)


def _generate_mock(
    request: GenerationRequest,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(request.output_path), fourcc, 16.0, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open output path: {request.output_path}"
        )
    rng = np.random.default_rng(request.seed + 1)
    colour = rng.integers(50, 200, size=3, dtype=np.uint8).tolist()
    frame = np.full((out_h, out_w, 3), colour, dtype=np.uint8)
    for _ in range(request.frames):
        writer.write(frame)
    writer.release()
    return GenerationResult(
        video_path=request.output_path,
        backend="mock-controlnet",
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _generate_vace_1b(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    _validate_vace_inputs(request)
    model_source = _resolve_model_source(config)

    try:
        import torch  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
        from diffusers import AutoencoderKLWan, WanVACEPipeline  # type: ignore[import]
        from diffusers.schedulers.scheduling_unipc_multistep import (  # type: ignore[import]
            UniPCMultistepScheduler,
        )
    except ImportError as exc:
        raise ImportError(
            "Wan VACE requires torch, Pillow, and diffusers.\n"
            "Run: pip install diffusers>=0.33 transformers accelerate pillow"
        ) from exc

    device = _resolve_device(config, torch)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    if getattr(torch.cuda, "is_available", lambda: False)():
        torch.cuda.empty_cache()

    vae = AutoencoderKLWan.from_pretrained(
        model_source,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(
        model_source,
        vae=vae,
        torch_dtype=dtype,
    )

    flow_shift = 5.0 if max(out_w, out_h) >= 720 else 3.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
    _apply_memory_policy(pipe, config, device)

    reference_image = Image.open(request.segmented_image_path).convert("RGBA")
    background = Image.new("RGBA", reference_image.size, (0, 0, 0, 255))
    reference_image = Image.alpha_composite(background, reference_image).convert("RGB")

    width, height = _snap_size(pipe, out_w, out_h)
    reference_image = reference_image.resize((width, height), Image.LANCZOS)
    conditioning_video = _load_video_frames(
        request.conditioning_video_path,
        mode="RGB",
        target_size=(width, height),
        expected_frames=request.frames,
    )
    conditioning_mask = _load_video_frames(
        request.conditioning_mask_path,
        mode="L",
        target_size=(width, height),
        expected_frames=request.frames,
    )

    prompt = (
        "Animate the reference character following the provided skeleton motion. "
        "Preserve character identity, keep anatomy clean, and maintain stable motion."
    )
    generator = torch.Generator(device=device).manual_seed(request.seed)
    output = pipe(
        video=conditioning_video,
        mask=conditioning_mask,
        reference_images=[reference_image],
        prompt=prompt,
        negative_prompt=_NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=request.frames,
        num_inference_steps=30,
        guidance_scale=5.0,
        generator=generator,
    ).frames[0]

    _write_output_video(request.output_path, output)

    del pipe, vae
    if getattr(torch.cuda, "is_available", lambda: False)():
        torch.cuda.empty_cache()

    return GenerationResult(
        video_path=request.output_path,
        backend="wan-1.3b-vace",
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _validate_vace_inputs(request: GenerationRequest) -> None:
    required_paths = [
        ("segmented image", request.segmented_image_path),
        ("conditioning video", request.conditioning_video_path),
        ("conditioning mask", request.conditioning_mask_path),
    ]
    for label, path in required_paths:
        if path is None:
            raise ValueError(f"VACE input {label} was not provided.")
        if not path.exists():
            raise FileNotFoundError(f"VACE input {label} not found: {path}")


def _resolve_model_source(config: MotionMirrorConfig) -> str:
    model_dir = config.model_cache("wan-1.3b-vace")
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Wan VACE weights not found in {model_dir}.\n"
            "Run: motion-mirror download --model wan-1.3b-vace"
        )
    if not (model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            f"Wan VACE weights in {model_dir} are incomplete.\n"
            "Expected a diffusers checkpoint with model_index.json."
        )
    return str(model_dir)


def _resolve_device(config: MotionMirrorConfig, torch: object) -> str:
    if config.device == "cuda" and getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    return "cpu"


def _apply_memory_policy(pipe: object, config: MotionMirrorConfig, device: str) -> None:
    if config.offload_model and device.startswith("cuda") and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe.to(device)

    if config.t5_cpu:
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "to"):
            text_encoder.to("cpu")


def _snap_size(pipe: object, out_w: int, out_h: int) -> tuple[int, int]:
    scale_factor = int(getattr(pipe, "vae_scale_factor_spatial", 8))
    patch_size = getattr(getattr(pipe, "transformer", None), "config", None)
    patch_value = getattr(patch_size, "patch_size", (1, 2))
    if isinstance(patch_value, (tuple, list)):
        patch_value = int(patch_value[1] if len(patch_value) > 1 else patch_value[0])
    else:
        patch_value = int(patch_value)

    mod_value = max(scale_factor * patch_value, 1)
    width = (out_w // mod_value) * mod_value or mod_value
    height = (out_h // mod_value) * mod_value or mod_value
    return width, height


def _load_video_frames(
    video_path: Path | None,
    mode: str,
    target_size: tuple[int, int],
    expected_frames: int,
) -> list[object]:
    if video_path is None:
        raise ValueError("Expected a conditioning video path.")

    from PIL import Image  # type: ignore[import]

    cap = cv2.VideoCapture(str(video_path))
    frames: list[object] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mode == "RGB":
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(converted, "RGB")
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pil = Image.fromarray(gray, "L")
            frames.append(pil.resize(target_size, Image.NEAREST))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Conditioning video produced no readable frames: {video_path}")
    if len(frames) != expected_frames:
        frames = _resample_frames(frames, expected_frames)
    return frames


def _resample_frames(frames: list[object], expected_frames: int) -> list[object]:
    indices = np.linspace(0, len(frames) - 1, expected_frames).round().astype(np.int32)
    return [frames[int(idx)] for idx in indices]


def _write_output_video(output_path: Path, frames: object) -> None:
    frames_list = frames if isinstance(frames, list) else list(frames)
    if not frames_list:
        raise RuntimeError("Wan VACE returned no frames")

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
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
