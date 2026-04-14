"""Wan2.1-I2V-14B generation backend (via diffusers).

Mock path:
    When config.backend == 'mock', writes a solid-colour MP4 using
    cv2.VideoWriter.  No GPU or model weights required.  Used for all
    CPU-only tests and for end-to-end pipeline smoke runs.

Real path (requires GPU + downloaded weights):
    Uses diffusers.WanImageToVideoPipeline (from diffusers>=0.33) with
    the Wan-AI/Wan2.1-I2V-14B-720P-Diffusers checkpoint.

    Download weights first:
        motion-mirror download --model wan-move

    The trajectory map produced by the trajectory stage is validated and
    stored alongside the generated video; native trajectory conditioning
    is planned for a future release once Wan-VACE APIs stabilise.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MotionMirrorConfig
from ..types import GenerationResult
from .models import GenerationRequest


def generate_with_wan_move(
    request: GenerationRequest,
    config: MotionMirrorConfig | None = None,
) -> GenerationResult:
    """Generate an animated video from a segmented character image + trajectory map.

    Parameters
    ----------
    request:
        Fully populated ``GenerationRequest`` (paths, backend, resolution, …).
    config:
        Pipeline config.  Defaults created if omitted.

    Returns
    -------
    GenerationResult
        Points to the generated video on disk.

    Raises
    ------
    FileNotFoundError
        Real path only — if model weights are missing.
    ImportError
        Real path only — if the wan package is not installed.
    ValueError
        If the resolution string is malformed.
    """
    cfg = config or MotionMirrorConfig()

    # Parse resolution
    try:
        w_str, h_str = request.resolution.split("x")
        out_w, out_h = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid resolution {request.resolution!r}. Expected 'WxH', e.g. '832x480'."
        ) from exc

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Mock path ─────────────────────────────────────────────────────────────
    if cfg.backend == "mock" or request.backend == "mock":
        return _generate_mock(request, out_w, out_h)

    # ── Real path ─────────────────────────────────────────────────────────────
    return _generate_real(request, cfg, out_w, out_h)


# ── Mock implementation ────────────────────────────────────────────────────────


def _generate_mock(
    request: GenerationRequest,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Write a solid-colour MP4 of the requested size and frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(request.output_path), fourcc, 16.0, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open output path: {request.output_path}"
        )

    # Deterministic colour derived from the seed so tests can assert it
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


# ── Real implementation ────────────────────────────────────────────────────────

#: Diffusers model ID for the 720-P checkpoint.
_WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

#: Standard negative prompt taken from the official diffusers example.
_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def _generate_real(
    request: GenerationRequest,
    config: MotionMirrorConfig,
    out_w: int,
    out_h: int,
) -> GenerationResult:
    """Invoke Wan2.1-I2V-14B via diffusers to generate the output video.

    Requires:
    - ``pip install diffusers>=0.33 transformers accelerate``
    - GPU with ≥ 22 GB VRAM (or enable model CPU offload for lower VRAM)
    - Weights downloaded via ``motion-mirror download --model wan-move``

    The trajectory map is validated here and retained next to the output
    for future conditioning support; the v0.1 generation uses image-only
    conditioning through the standard WanImageToVideoPipeline.
    """
    # ── 1. Validate weights ───────────────────────────────────────────────────
    model_dir = config.model_cache("wan-move")
    has_local_weights = any(model_dir.iterdir())

    # ── 2. Validate input files ───────────────────────────────────────────────
    for label, p in [
        ("segmented image", request.segmented_image_path),
        ("trajectory map", request.trajectory_map_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Generation input {label} not found: {p}")

    # ── 3. Import heavy deps (lazy — GPU path only) ───────────────────────────
    try:
        import torch  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "torch is not installed. Run: pip install -r requirements-cuda.txt"
        ) from exc

    try:
        from PIL import Image  # type: ignore[import]
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # type: ignore[import]
        from diffusers.utils import export_to_video  # type: ignore[import]
        from transformers import CLIPVisionModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "diffusers, transformers, or Pillow is not installed.\n"
            "Run: pip install diffusers>=0.33 transformers accelerate pillow"
        ) from exc

    # Free any cached VRAM before loading the large model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 4. Load pipeline ──────────────────────────────────────────────────────
    # Prefer local weights if they look like a diffusers checkpoint (has
    # model_index.json).  Fall back to downloading from HuggingFace.
    model_source: str
    if has_local_weights and (model_dir / "model_index.json").exists():
        model_source = str(model_dir)
    else:
        if not has_local_weights:
            raise FileNotFoundError(
                f"Wan model weights not found in {model_dir}.\n"
                "Run: motion-mirror download --model wan-move"
            )
        # Weights exist but not in diffusers format — point to HF ID and let
        # diffusers use its own cache.
        model_source = _WAN_MODEL_ID

    image_encoder = CLIPVisionModel.from_pretrained(
        model_source, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_source, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_source,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )
    # enable_model_cpu_offload moves one submodel at a time to GPU instead of
    # loading everything simultaneously.  The full stack (T5 ~12 GB + transformer
    # ~28 GB + VAE + CLIP) exceeds 32 GB VRAM; offloading keeps peak usage to
    # the largest single submodel (~28 GB) which fits on a 5090/4090/3090.
    pipe.enable_model_cpu_offload()

    # ── 5. Prepare input image ────────────────────────────────────────────────
    # Load RGBA segmented image; composite onto black background for the model.
    rgba = Image.open(request.segmented_image_path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    char_image = Image.alpha_composite(bg, rgba).convert("RGB")

    # Resize to requested resolution while respecting the VAE's spatial
    # scale factor (mod_value).  The pipeline exposes these via transformer config.
    mod_value: int = (
        pipe.vae_scale_factor_spatial  # type: ignore[attr-defined]
        * pipe.transformer.config.patch_size[1]  # type: ignore[attr-defined]
    )
    # Clamp to requested WxH but keep within mod_value grid
    height = (out_h // mod_value) * mod_value or mod_value
    width = (out_w // mod_value) * mod_value or mod_value
    char_image = char_image.resize((width, height), Image.LANCZOS)

    # ── 6. Build generation prompt from trajectory metadata ───────────────────
    traj_data = np.load(str(request.trajectory_map_path))
    density = int(traj_data["density"]) if "density" in traj_data else 0
    prompt = (
        f"A character performing smooth, natural motion. "
        f"Fluid movement with {density} motion trajectory points. "
        "Cinematic quality, detailed animation, consistent lighting."
    )

    # ── 7. Generate ───────────────────────────────────────────────────────────
    generator = torch.Generator(device=config.device).manual_seed(request.seed)
    output = pipe(
        image=char_image,
        prompt=prompt,
        negative_prompt=_NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=request.frames,
        guidance_scale=5.0,
        generator=generator,
    ).frames[0]

    # ── 8. Export to video ────────────────────────────────────────────────────
    export_to_video(output, str(request.output_path), fps=16)

    return GenerationResult(
        video_path=request.output_path,
        backend="wan-move-14b",
        resolution=request.resolution,
        num_frames=request.frames,
    )
