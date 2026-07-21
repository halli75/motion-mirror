<div align="center">

# Motion Mirror

**Local-first motion transfer - animate any character image from a reference video.**

[![CI](https://github.com/halli75/motion-mirror/actions/workflows/ci.yml/badge.svg)](https://github.com/halli75/motion-mirror/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Give Motion Mirror a character image and a reference video; it produces a video of your character performing the same motion. Everything runs on your machine - no cloud, no API keys, no per-clip fees.

An open-source alternative to hosted motion-control tools, built on [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1).

</div>

---

## Examples

<!-- EXAMPLES: side-by-side character / reference / result clips go here -->
*Example generations coming soon.*

---

## Features

- **Runs locally** - from 9 GB VRAM (1.3B backend) up to full 14B quality on consumer cards
- **Whole-body motion** - 133-keypoint skeleton conditioning (body + hands + face) markedly reduces facial and hand flicker
- **Multiple backends** - pick speed (1.3B), identity fidelity (14B GGUF), or full precision (14B); `--auto` selects for your VRAM
- **CLI, Python API, and web UI** - plus ComfyUI custom nodes
- **Audio passthrough** - the reference video's audio lands in the result

## Quick start

```bash
# PyTorch with CUDA (match your driver; cu124 shown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Motion Mirror + GPU inference deps
pip install -e ".[cuda,gpu-inference]"

# Model weights → ~/.cache/motion-mirror/
motion-mirror download --model dwpose           # DWPose-L pose (~350 MB)
motion-mirror download --model wan-1.3b-vace    # Wan2.1-VACE-1.3B (~19 GB)

# Animate!
motion-mirror run character.png motion.mp4 --backend wan-1.3b-vace --offload-model --t5-cpu
```

Output lands in `./outputs/result.mp4`. For the best identity fidelity, use the quantized 14B backend:

```bash
motion-mirror download --model vace-14b-gguf    # Q4_K_M + base (~24 GB)
motion-mirror run character.png motion.mp4 \
  --backend wan-14b-vace-gguf --frames 81 --steps 50 --resolution 480x832
```

Or launch the web UI:

```bash
motion-mirror ui
```

Windows users: see [docs/windows-install.md](docs/windows-install.md). Optional SAM-2 segmenter: `pip install "git+https://github.com/facebookresearch/sam2.git"`.

## How it works

```
character.png + motion.mp4
        │
   [1] Segment      rembg / SAM-2 remove the character's background → RGBA
   [2] Extract pose DWPose-L detects 133 whole-body keypoints per frame
   [3] Condition    render the canonical OpenPose whole-body skeleton
                    (body + hands + face) + mask frames for VACE
   [4] Generate     Wan2.1-VACE renders the character following the skeleton
   [5] Mux audio    original audio passed through
        │
        ▼
     result.mp4
```

Rendering the **whole-body** skeleton - not just the 18 body joints - gives VACE the hand and face structure it needs, which markedly reduces facial/hand flicker in the output.

## Requirements

| Component | `wan-1.3b-vace` | `wan-14b-vace-gguf` |
|---|---|---|
| GPU VRAM (free) | ~9 GB | ~18 GB |
| System RAM | 32 GB | 40 GB |
| Disk (model cache) | ~20 GB | ~45 GB |
| CUDA / driver | 12.x / 570+ | 12.x / 570+ |
| Python | 3.11+ | 3.11+ |

CPU-only real generation is not supported (the `mock` backend works for testing without a GPU).

## Backends

| Backend | VRAM (measured) | Identity | Notes |
|---|---:|---|---|
| `wan-1.3b-vace` | 8.0 GB | loose | Fast, lightest. Follows motion well; face/appearance can drift. |
| `wan-14b-vace-gguf` | ~18 GB | strong | **Recommended.** Q4_K_M quantized 14B - locks identity, faster than the full 14B. |
| `wan-14b-vace` | 8.0 GB* | strong | Full 14B. *Fits ~9 GB via sequential CPU offload, but slower. |
| `mock` | - | - | Solid-colour output for testing without a GPU. |

All backends are GPU-validated end-to-end (RTX 3090/4090/A6000). `--auto` selects `wan-1.3b-vace` and never routes to a 14B backend implicitly.

## Fast generation

`--fast` swaps in a community-distilled few-step artifact per backend: ~5-10x fewer denoising steps with modest quality loss (reduced motion amplitude, slight control-adherence loss possible). Classifier-free guidance is disabled (itself ~2x compute saving).

```bash
motion-mirror download --model fast   # 14B distill LoRA, ~631 MB
motion-mirror run character.png motion.mp4 --backend wan-14b-vace --fast
```

| Backend | Fast artifact | Steps | License |
|---|---|---:|---|
| `wan-14b-vace` | LightX2V step/CFG distill LoRA (~631 MB) | 4 | Apache 2.0 |
| `wan-1.3b-vace` | Self-Forcing DMD LoRA (~91 MB) | 4 | **CC-BY-NC-SA-4.0 - non-commercial only** |
| `wan-14b-vace-gguf` | FusionX pre-merged distilled GGUF Q4_K_M (~11.6 GB) | 8 | Apache 2.0 (⚠️ experimental) |

- Explicit `--steps` / `--guidance-scale` always override the fast defaults. If control fidelity suffers, try 6-8 steps.
- **1.3B fast is non-commercial**: the Self-Forcing weights are CC-BY-NC-SA-4.0, so outputs generated with them cannot be used commercially. The artifact is excluded from the `fast` download group - fetch it explicitly with `motion-mirror download --model wan-fast-1.3b` (a license warning prints at download and generation time).
- **FusionX GGUF fast is experimental**: validated in ComfyUI only, untested via diffusers until the next GPU validation run. Fetch with `motion-mirror download --model wan-14b-vace-fusionx-gguf`.

**Bring your own LoRA:** `--lora` accepts a local `.safetensors` path, a Hugging Face repo id, or `repo_id:filename`, with `--lora-scale` controlling fuse strength (default 1.0). The LoRA is fused into the transformer before generation (zero runtime overhead). Not supported on GGUF backends (diffusers limitation), and mutually exclusive with `--fast`.

```bash
motion-mirror run character.png motion.mp4 --backend wan-14b-vace \
  --lora someuser/some-wan-lora:pytorch_lora_weights.safetensors --lora-scale 0.8
```

## CLI reference

```
motion-mirror run        Run the full pipeline
motion-mirror download   Fetch model weights
motion-mirror presets    List generation presets
motion-mirror benchmark  System / GPU diagnostics
motion-mirror ui         Launch the Gradio UI
```

Key `run` options:

| Option | Meaning |
|---|---|
| `--backend` | `auto` \| `wan-1.3b-vace` \| `wan-14b-vace` \| `wan-14b-vace-gguf` \| `mock` |
| `--frames` | output frame count (default 81 ≈ 5 s @ 16 fps) |
| `--steps` | denoising steps, 1–200 (default 30, or 4-8 with `--fast`; 50 for higher quality) |
| `--fast` | distilled few-step generation (see [Fast generation](#fast-generation)) |
| `--guidance-scale` | CFG scale, > 0 (default 5.0; 1.0 = CFG off) |
| `--lora` / `--lora-scale` | fuse a custom LoRA (path, repo id, or `repo_id:filename`) |
| `--resolution` | `WxH`, e.g. `480x832` (portrait) or `832x480` |
| `--offload-model` / `--t5-cpu` | trade speed for VRAM |
| `--segmenter` | `rembg` (default) \| `sam2` |
| `--flow-estimator` | `farneback` (default) \| `raft` |

`motion-mirror download --model` also accepts groups: `dwpose`, `vace`, `vace-14b`, `vace-14b-gguf`, `extras`, `fast`, `all`. A disk-space check runs before each download; `all` deliberately excludes the large 14B groups - request those by name. `fast` is the Apache 14B LoRA only; the non-commercial 1.3B artifact and the ~11.6 GB FusionX GGUF are explicit-name-only.

Presets: `default`, `fast`, `low-vram`, `mock` - see `motion-mirror presets --list`.

## Python API

```python
from pathlib import Path
from motion_mirror import MotionMirrorPipeline, MotionMirrorConfig

cfg = MotionMirrorConfig(
    backend="wan-14b-vace-gguf",
    resolution="480x832",
    num_frames=81,
    num_inference_steps=50,
    offload_model=True,
    t5_cpu=True,
)
result = MotionMirrorPipeline(cfg).run(
    image_path=Path("character.png"),
    motion_video_path=Path("motion.mp4"),
)
print(result.output_path)  # → outputs/result.mp4
```

Typed exceptions (all inherit `MotionMirrorError`): `NoPoseDetectedError`, `MultiplePeopleDetectedError`, `SmallSubjectError`, `SmallSubjectWarning`, `UnsupportedImageError`, `UnsupportedVideoError`, `VideoDecodeError`, `MultipleCharactersError`.

## Known limitations

- **Single-person only** - multi-person reference videos raise `MultiplePeopleDetectedError`. Low-confidence background detections are filtered automatically; crop to one clear subject.
- **Background** - the output background follows the character image, not the reference video's scene.
- **Generation time** - under CPU offload a full 81-frame clip takes tens of minutes; much faster on 24 GB+ cards.
- **Drivers** - needs an NVIDIA driver new enough for your PyTorch CUDA build (570/CUDA 12.8 validated).

## Contributing

Contributions are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test markers, and PR conventions.

```bash
pip install -e ".[dev]"
pytest -m "not gpu"     # fast suite, runs in CI
pytest -m gpu           # needs CUDA + weights
```

GPU validation is reproducible via [`runpod-validation/`](runpod-validation/README.md) (API-only RunPod harness, no SSH).

## License

[MIT](LICENSE) - model weights and dependencies keep their own licenses; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Generated outputs are subject to the Wan2.1 model terms (Apache 2.0, commercial use permitted). Exception: `--fast` on the 1.3B backend uses a CC-BY-NC-SA-4.0 distill LoRA - those outputs are **non-commercial only**.

## Acknowledgments

Motion Mirror stands on the shoulders of [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) (video generation), [DWPose](https://github.com/IDEA-Research/DWPose) / [rtmlib](https://github.com/Tau-J/rtmlib) (pose estimation), [rembg](https://github.com/danielgatis/rembg) and [SAM-2](https://github.com/facebookresearch/sam2) (segmentation), and [🤗 Diffusers](https://github.com/huggingface/diffusers).
