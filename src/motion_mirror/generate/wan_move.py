"""Wan image-to-video generation backends.

Supported backends
------------------
mock
    Writes a solid-colour MP4 using cv2.VideoWriter. No GPU or model weights
    required. Used for CPU-only tests and pipeline smoke runs.

wan-move-14b
    Uses diffusers.WanImageToVideoPipeline with the
    Wan-AI/Wan2.1-I2V-14B-720P-Diffusers checkpoint.

wan-move-fast
    Uses the LightX2V runtime with a Wan2.1 four-step distilled model plus the
    companion Wan components (T5, CLIP, VAE, tokenizers, config) arranged in a
    dedicated fast-model cache directory.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest

_WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

_FAST_DIT_CANDIDATES: tuple[str, ...] = (
    "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "wan2.1_i2v_720p_lightx2v_4step.safetensors",
    "wan2.1_i2v_720p_int8_lightx2v_4step.safetensors",
    "wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "wan2.1_i2v_480p_lightx2v_4step.safetensors",
    "wan2.1_i2v_480p_int8_lightx2v_4step.safetensors",
)
_FAST_T5_CANDIDATES: tuple[str, ...] = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_t5_umt5-xxl-enc-fp8.pth",
)
_FAST_CLIP_CANDIDATES: tuple[str, ...] = (
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth",
)
_FAST_CONFIG_CANDIDATES: tuple[str, ...] = ("config.json", "configuration.json")
_FAST_RUNTIME_CONFIG_NAME = "wan_i2v_distill_4step_cfg_4090.json"
_FAST_RUNTIME_CONFIG_TEMPLATE: dict[str, object] = {
    "cpu_offload": True,
    "offload_granularity": "block",
    "t5_cpu_offload": False,
    "vae_cpu_offload": False,
    "clip_cpu_offload": False,
    "self_attn_1_type": "sage_attn2",
    "cross_attn_1_type": "sage_attn2",
    "cross_attn_2_type": "sage_attn2",
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "infer_steps": 4,
    "sample_guide_scale": 5.0,
    "sample_shift": 5.0,
    "enable_cfg": False,
    "denoising_step_list": [1000, 750, 500, 250],
    "dit_quantized": True,
    "dit_quant_scheme": "fp8-q8f",
    "t5_quantized": False,
    "t5_quant_scheme": "fp8-q8f",
    "clip_quantized": False,
    "clip_quant_scheme": "fp8-q8f",
}


def generate_with_wan_move(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate an animated video from a segmented character image."""
    cfg = config or MotionMirrorConfig()
    out_w, out_h = _parse_resolution(request.resolution)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "mock" or request.backend == "mock":
        return _generate_mock(request, out_w, out_h)
    if cfg.backend == "wan-move-fast" or request.backend == "wan-move-fast":
        return _generate_fast(request, cfg, out_w, out_h)
    return _generate_real(request, cfg, out_w, out_h)


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        w_str, h_str = resolution.split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {resolution!r}. Expected 'WxH', e.g. '832x480'."
        ) from exc


def _generate_mock(
    request: GenerationRequest,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Write a solid-colour MP4 of the requested size and frame count."""
    writer = cv2.VideoWriter(
        str(request.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        16.0,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open output path: {request.output_path}"
        )

    rng = np.random.default_rng(request.seed)
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


def _generate_real(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Invoke Wan2.1-I2V-14B via diffusers."""
    _validate_common_inputs(request)
    model_source = _resolve_wan_model_source(config)

    try:
        import torch  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "torch is not installed. Run: pip install -r requirements-cuda.txt"
        ) from exc

    try:
        from PIL import Image  # type: ignore[import]
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # type: ignore[import]
        from transformers import CLIPVisionModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "diffusers, transformers, or Pillow is not installed.\n"
            "Run: pip install diffusers>=0.33 transformers accelerate pillow"
        ) from exc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_encoder = CLIPVisionModel.from_pretrained(
        model_source,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_source,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_source,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )
    # Apply VRAM strategy based on config flags
    if config.offload_model:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(config.device)
    if config.t5_cpu:
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "to"):
            text_encoder.to("cpu")
    pipe.enable_attention_slicing(1)

    char_image = _load_character_image(request.segmented_image_path)
    width, height = _snap_wan_size(pipe, out_w, out_h)
    char_image = char_image.resize((width, height), Image.LANCZOS)

    generator = torch.Generator(device=config.device).manual_seed(request.seed)
    output = pipe(
        image=char_image,
        prompt=_build_prompt(request.trajectory_map_path),
        negative_prompt=_NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=request.frames,
        guidance_scale=5.0,
        generator=generator,
    ).frames[0]

    _write_output_frames(request.output_path, output)

    del pipe, vae, image_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return GenerationResult(
        video_path=request.output_path,
        backend="wan-move-14b",
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _generate_fast(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Invoke Wan2.1 four-step distilled I2V via LightX2V."""
    _validate_common_inputs(request)
    model_dir = _resolve_lightx2v_model_dir(config)
    runtime_config_path = _build_lightx2v_runtime_config(
        model_dir=model_dir,
        request=request,
        config=config,
        out_w=out_w,
        out_h=out_h,
    )
    prepared_image_path = _prepare_lightx2v_input_image(
        request.segmented_image_path,
        request.output_path,
        (out_w, out_h),
    )

    try:
        from lightx2v import LightX2VPipeline  # type: ignore[import]
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        detail = f" Missing dependency: {missing}." if missing else ""
        raise ImportError(
            "LightX2V fast backend requires the LightX2V runtime and its "
            f"transitive dependencies.{detail}\n"
            "Run: pip install -e '.[lightx2v]'"
        ) from exc

    pipe = LightX2VPipeline(
        model_path=str(model_dir),
        model_cls="wan2.1",
        task="i2v",
    )

    if config.offload_model or config.t5_cpu:
        if not hasattr(pipe, "enable_offload"):
            raise RuntimeError(
                "Installed LightX2V runtime does not expose enable_offload(), "
                "but offload flags were requested."
            )
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=(config.offload_model or config.t5_cpu),
            image_encoder_offload=False,
            vae_offload=False,
        )

    pipe.create_generator(config_json=str(runtime_config_path))

    pipe.generate(
        seed=request.seed,
        image_path=str(prepared_image_path),
        prompt=_build_prompt(request.trajectory_map_path),
        negative_prompt="",
        save_result_path=str(request.output_path),
    )

    if not request.output_path.exists():
        raise RuntimeError(
            f"LightX2V did not produce an output video at {request.output_path}"
        )

    _empty_torch_cache_if_available()

    return GenerationResult(
        video_path=request.output_path,
        backend="wan-move-fast",
        resolution=request.resolution,
        num_frames=request.frames,
    )


def _validate_common_inputs(request: GenerationRequest) -> None:
    for label, path in (
        ("segmented image", request.segmented_image_path),
        ("trajectory map", request.trajectory_map_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Generation input {label} not found: {path}")


def _resolve_wan_model_source(config: MotionMirrorConfig) -> str:
    model_dir = config.model_cache("wan-move")
    has_local_weights = model_dir.exists() and any(model_dir.iterdir())
    if has_local_weights and (model_dir / "model_index.json").exists():
        return str(model_dir)
    if not has_local_weights:
        raise FileNotFoundError(
            f"Wan model weights not found in {model_dir}.\n"
            "Run: motion-mirror download --model wan-move"
        )
    return _WAN_MODEL_ID


def _resolve_lightx2v_model_dir(config: MotionMirrorConfig) -> Path:
    model_dir = config.model_cache("wan-move-fast")
    ensure_lightx2v_fast_configs(model_dir)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"LightX2V fast model assets not found in {model_dir}.\n"
            "Run: motion-mirror download --model fast"
        )

    missing: list[str] = []
    if _find_existing_path(model_dir, _FAST_DIT_CANDIDATES) is None:
        missing.append("one LightX2V Wan2.1 I2V 4-step .safetensors file")
    if _find_existing_path(model_dir, _FAST_T5_CANDIDATES) is None:
        missing.append("T5 encoder checkpoint")
    if _find_existing_path(model_dir, _FAST_CLIP_CANDIDATES) is None:
        missing.append("CLIP encoder checkpoint")
    if not (model_dir / "Wan2.1_VAE.pth").exists():
        missing.append("Wan2.1_VAE.pth")
    if _find_existing_path(model_dir, _FAST_CONFIG_CANDIDATES) is None:
        missing.append("config.json")
    if not (model_dir / "google").exists():
        missing.append("google tokenizer directory")
    if not (model_dir / "xlm-roberta-large").exists():
        missing.append("xlm-roberta-large tokenizer directory")

    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"LightX2V fast backend is missing required assets in {model_dir}: {joined}.\n"
            "Run: motion-mirror download --model fast"
        )

    return model_dir


def ensure_lightx2v_fast_configs(model_dir: Path) -> None:
    """Materialize the bundled 24 GB LightX2V distill template in the cache."""
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / _FAST_RUNTIME_CONFIG_NAME
    if config_path.exists():
        return
    config_path.write_text(
        json.dumps(_FAST_RUNTIME_CONFIG_TEMPLATE, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_lightx2v_runtime_config(
    model_dir: Path,
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> Path:
    ensure_lightx2v_fast_configs(model_dir)
    template_path = model_dir / _FAST_RUNTIME_CONFIG_NAME
    runtime_config = json.loads(template_path.read_text(encoding="utf-8"))

    attn_mode = _resolve_lightx2v_attention_backend()
    text_offload = config.offload_model or config.t5_cpu
    runtime_config.update(
        {
            "target_video_length": request.frames,
            "target_height": out_h,
            "target_width": out_w,
            "cpu_offload": bool(text_offload),
            "offload_granularity": "block",
            "t5_cpu_offload": bool(text_offload),
            "vae_cpu_offload": False,
            "clip_cpu_offload": False,
            "self_attn_1_type": attn_mode,
            "cross_attn_1_type": attn_mode,
            "cross_attn_2_type": attn_mode,
        }
    )

    runtime_config_path = request.output_path.with_name("lightx2v_runtime_config.json")
    runtime_config_path.write_text(
        json.dumps(runtime_config, indent=2) + "\n",
        encoding="utf-8",
    )
    return runtime_config_path


def _resolve_lightx2v_attention_backend() -> str:
    if _module_exists("sageattention") or _module_exists("sageattn"):
        return "sage_attn2"
    if _module_exists("flash_attn"):
        return "flash_attn2"
    return "torch"


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _find_existing_path(base_dir: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    return None


def _load_character_image(segmented_image_path: Path) -> "Image.Image":
    from PIL import Image  # type: ignore[import]

    rgba = Image.open(segmented_image_path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def _prepare_lightx2v_input_image(
    segmented_image_path: Path,
    output_path: Path,
    size: tuple[int, int],
) -> Path:
    prepared_path = output_path.with_name("lightx2v_input.png")
    image = _load_character_image(segmented_image_path)
    image = image.resize(size)
    image.save(prepared_path)
    return prepared_path


def _snap_wan_size(pipe: object, out_w: int, out_h: int) -> tuple[int, int]:
    mod_value = (
        pipe.vae_scale_factor_spatial  # type: ignore[attr-defined]
        * pipe.transformer.config.patch_size[1]  # type: ignore[attr-defined]
    )
    height = (out_h // mod_value) * mod_value or mod_value
    width = (out_w // mod_value) * mod_value or mod_value
    return width, height


def _build_prompt(trajectory_map_path: Path) -> str:
    # TODO(v0.3): Inject trajectory data directly into the diffusion process
    # as spatial conditioning rather than as prompt text.  This requires the
    # Wan-Move fine-tuned checkpoint (not vanilla Wan2.1-I2V) and a conditioned
    # pipeline call that accepts trajectory tensors.  Until those weights are
    # available the trajectory density is reflected in the text prompt only.
    traj_data = np.load(str(trajectory_map_path))
    density = int(traj_data["density"]) if "density" in traj_data else 0
    return (
        f"A character performing smooth, natural motion. "
        f"Fluid movement with {density} motion trajectory points. "
        "Cinematic quality, detailed animation, consistent lighting."
    )


def _write_output_frames(output_path: Path, frames: object) -> None:
    frames_list = frames if isinstance(frames, list) else list(frames)
    if not frames_list:
        raise RuntimeError("Pipeline returned no frames")

    first_frame = np.array(frames_list[0])
    height, width = first_frame.shape[:2]
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


def _empty_torch_cache_if_available() -> None:
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
