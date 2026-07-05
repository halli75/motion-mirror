"""Wan2.1 VACE generation backend."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest

# Per-backend model specs. `name` echoes the backend key so the result can
# report the ACTUAL backend used. `download_group` names the CLI group
# (`motion-mirror download --model <group>`) that fetches this backend's
# weights, so error hints point at the right download. GGUF backends carry a
# `gguf_filename` (the quantized transformer) plus `base_cache_subdir` /
# `full_cache_subdir` for the un-quantized base components (VAE, text encoder,
# scheduler, config).
_VACE_BACKEND_SPECS: dict[str, dict] = {
    "wan-1.3b-vace": {
        "name": "wan-1.3b-vace",
        "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "cache_subdir": "wan-1.3b-vace",
        "download_group": "vace",
        "gguf": None,
    },
    "wan-14b-vace": {
        "name": "wan-14b-vace",
        "model_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "cache_subdir": "wan-14b-vace",
        "download_group": "vace-14b",
        "gguf": None,
    },
    "wan-14b-vace-gguf": {
        "name": "wan-14b-vace-gguf",
        "model_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "cache_subdir": "wan-14b-vace-gguf",
        "base_cache_subdir": "wan-14b-vace-base",
        "full_cache_subdir": "wan-14b-vace",
        "gguf_filename": "Wan2.1_14B_VACE-Q4_K_M.gguf",
        "download_group": "vace-14b-gguf",
    },
}
# Describe the desired subject, NEVER the conditioning mechanism: the text
# prompt dominates subject choice, and naming the control signal ("skeleton")
# made VACE render an anatomical skeleton instead of the reference character
# (Wave-2 GPU run, 2026-07-03).
_VACE_PROMPT = (
    "A person performs a smooth, continuous dance in a well-lit space. "
    "The person's appearance, face, hairstyle, and clothing exactly match "
    "the reference image. Photorealistic, natural lighting, high detail, "
    "stable camera, clean anatomy."
)
_NEGATIVE_PROMPT = (
    "skeleton, bones, x-ray, anatomical model, stick figure, line drawing, "
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def generate_with_vace(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate via a Wan VACE backend (1.3B, 14B, or 14B-GGUF)."""
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

    # Prefer the request's explicit backend; fall back to config for
    # placeholder values ("auto"/"mock"/unset).
    backend = request.backend if request.backend not in (None, "auto", "mock") else cfg.backend
    spec = _VACE_BACKEND_SPECS.get(backend)
    if spec is None:
        raise ValueError(
            f"Unknown VACE backend {backend!r}. "
            f"Known VACE backends: {sorted(_VACE_BACKEND_SPECS)}."
        )

    return _generate_vace(request, cfg, out_w, out_h, spec)


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
        backend="mock",
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _generate_vace(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
    spec: dict,
) -> GenerationResult:
    _validate_vace_inputs(request)
    is_gguf = spec.get("gguf_filename") is not None

    if is_gguf:
        transformer_path = _resolve_gguf_transformer_path(config, spec)
    else:
        model_source = _resolve_model_source(config, spec)

    try:
        import torch  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
        if is_gguf:
            from diffusers import (  # type: ignore[import]
                AutoencoderKLWan,
                GGUFQuantizationConfig,
                WanVACEPipeline,
                WanVACETransformer3DModel,
            )
        else:
            from diffusers import AutoencoderKLWan, WanVACEPipeline  # type: ignore[import]
        from diffusers.schedulers.scheduling_unipc_multistep import (  # type: ignore[import]
            UniPCMultistepScheduler,
        )
        if is_gguf:
            # diffusers defers the gguf-package check until dequantization;
            # import it here so a missing dep fails with THIS message instead
            # of a diffusers-internal error mid-load.
            import gguf  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Wan VACE requires torch, Pillow, and diffusers>=0.35 "
            "(plus the gguf package for the GGUF backend).\n"
            'Run: pip install "diffusers>=0.35.0" transformers accelerate pillow gguf'
        ) from exc

    device = _resolve_device(config, torch)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    if getattr(torch.cuda, "is_available", lambda: False)():
        torch.cuda.empty_cache()

    transformer = None
    if is_gguf:
        base_source = _resolve_gguf_base_source(config, spec)
        transformer = WanVACETransformer3DModel.from_single_file(
            str(transformer_path),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=spec["model_id"],
            subfolder="transformer",
            torch_dtype=dtype,
        )
        # fp32 VAE avoids the bf16 decode artifacts seen on the non-quantized path.
        vae = AutoencoderKLWan.from_pretrained(
            base_source,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanVACEPipeline.from_pretrained(
            base_source,
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype,
        )
    else:
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
    _apply_memory_policy(pipe, config, device, gguf=is_gguf)

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

    generator = torch.Generator(device=device).manual_seed(request.seed)
    try:
        output = pipe(
            video=conditioning_video,
            mask=conditioning_mask,
            reference_images=[reference_image],
            prompt=_VACE_PROMPT,
            negative_prompt=_NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=request.frames,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=generator,
        ).frames[0]

        _write_output_video(request.output_path, output)
    finally:
        del pipe, vae
        # GGUF path holds an extra transformer handle; release it before
        # clearing the CUDA cache so the freed VRAM is actually reclaimed.
        if transformer is not None:
            del transformer
        if getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.empty_cache()

    return GenerationResult(
        video_path=request.output_path,
        backend=spec["name"],
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

    # Wan VACE's temporal VAE compresses in groups of 4 frames plus a leading
    # keyframe, so num_frames must satisfy (n - 1) % 4 == 0 (e.g. 81, 61, 17).
    if (request.frames - 1) % 4 != 0:
        raise ValueError(
            f"VACE requires num_frames of the form 4k+1 (e.g. 17, 61, 81); "
            f"got {request.frames}."
        )


def _resolve_model_source(config: MotionMirrorConfig, spec: dict) -> str:
    model_dir = config.model_cache(spec["cache_subdir"])
    group = spec["download_group"]
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Wan VACE weights not found in {model_dir}.\n"
            f"Run: motion-mirror download --model {group}"
        )
    if not (model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            f"Wan VACE weights in {model_dir} are incomplete.\n"
            "Expected a diffusers checkpoint with model_index.json.\n"
            f"Run: motion-mirror download --model {group}"
        )
    return str(model_dir)


def _resolve_gguf_base_source(config: MotionMirrorConfig, spec: dict) -> str:
    """Locate the un-quantized base components for the GGUF pipeline.

    Absent (nothing downloaded) and incomplete (a partial cache) are distinct:
    the first is a legitimate first-run that pulls from the hub, the second is
    a broken download the user must repair.
    """
    group = spec["download_group"]

    # (a) A full 14B checkpoint already on disk provides every base component.
    full_dir = config.model_cache(spec["full_cache_subdir"])
    if (full_dir / "model_index.json").exists():
        return str(full_dir)

    # (b) A dedicated base-components cache (VAE/text-encoder/config only).
    base_dir = config.model_cache(spec["base_cache_subdir"])
    if (base_dir / "model_index.json").exists():
        return str(base_dir)

    # (c) Base dir absent or empty: first-time run, stream from the hub.
    if not base_dir.exists() or not any(base_dir.iterdir()):
        print(
            f"[motion-mirror] Base components for {spec['name']} not cached; "
            f"a multi-GB download from {spec['model_id']} will start now."
        )
        return spec["model_id"]

    # (d) Base dir populated but missing model_index.json: partial/broken cache.
    raise FileNotFoundError(
        f"GGUF base components in {base_dir} are incomplete "
        "(missing model_index.json).\n"
        f"Run: motion-mirror download --model {group}"
    )


def _resolve_gguf_transformer_path(config: MotionMirrorConfig, spec: dict) -> Path:
    transformer_path = config.model_cache(spec["cache_subdir"]) / spec["gguf_filename"]
    if not transformer_path.exists() or transformer_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"GGUF transformer not found: {transformer_path}.\n"
            f"Run: motion-mirror download --model {spec['download_group']}"
        )
    return transformer_path


def _resolve_device(config: MotionMirrorConfig, torch: object) -> str:
    if config.device == "cuda" and getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    return "cpu"


def _apply_memory_policy(
    pipe: object,
    config: MotionMirrorConfig,
    device: str,
    *,
    gguf: bool = False,
) -> None:
    if (
        gguf
        and config.offload_model
        and device.startswith("cuda")
        and hasattr(pipe, "enable_model_cpu_offload")
    ):
        # GGUF-quantized params carry a `quant_type` attribute lost when
        # sequential offload round-trips each weight through the meta device
        # ... later crashing in diffusers' GGUF utils with `KeyError: None`.
        # Whole-module offload keeps each component intact on the CPU<->GPU
        # hops, preserving quant metadata.
        pipe.enable_model_cpu_offload()
        return

    if config.offload_model and device.startswith("cuda") and hasattr(pipe, "enable_sequential_cpu_offload"):
        # Sequential offload owns every submodule's device placement (weights
        # become meta tensors behind hooks); a manual t5_cpu move afterwards
        # raises "Cannot copy out of meta tensor".
        pipe.enable_sequential_cpu_offload()
        return

    if hasattr(pipe, "to"):
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
                gray = np.where(gray > 127, np.uint8(255), np.uint8(0))
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
