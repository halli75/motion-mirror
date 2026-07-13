# Motion Mirror

**Local-first motion transfer — animate any character image from a reference video.**

Give Motion Mirror a character image and a reference video; it produces a video of your character performing the same motion. Everything runs on your machine — no cloud, no API keys, no per-clip fees. It's an open-source alternative to hosted motion-control tools, built on [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1).

---

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

Rendering the **whole-body** skeleton — not just the 18 body joints — gives VACE the hand and face structure it needs, which markedly reduces facial/hand flicker in the output.

---

## Requirements

| Component | `wan-1.3b-vace` | `wan-14b-vace-gguf` |
|---|---|---|
| GPU VRAM (free) | ~9 GB | ~18 GB |
| System RAM | 32 GB | 40 GB |
| Disk (model cache) | ~20 GB | ~45 GB |
| CUDA / driver | 12.x / 570+ | 12.x / 570+ |
| Python | 3.11+ | 3.11+ |

CPU-only real generation is not supported (`mock` backend works for testing without a GPU).

---

## Installation

```bash
# 1. PyTorch with CUDA (match your driver; cu124 shown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Motion Mirror + GPU inference deps
pip install -e ".[cuda,gpu-inference]"

# 3. (optional) SAM-2 character segmenter
pip install "git+https://github.com/facebookresearch/sam2.git"

# 4. Model weights → ~/.cache/motion-mirror/
motion-mirror download --model dwpose          # DWPose-L pose (~350 MB)
motion-mirror download --model wan-1.3b-vace    # Wan2.1-VACE-1.3B (~19 GB)
motion-mirror download --model vace-14b-gguf    # Wan2.1-VACE-14B Q4_K_M + base (~24 GB)
```

`--model` also accepts groups: `dwpose`, `vace`, `vace-14b`, `vace-14b-gguf`, `extras`, `all`. A disk-space check runs before each download. `all` deliberately excludes the large 14B groups — request those by name.

---

## Quick start

```bash
# 1.3B (lightest, ~9 GB VRAM)
motion-mirror run character.png motion.mp4 --backend wan-1.3b-vace --offload-model --t5-cpu

# 14B GGUF (best identity, ~18 GB VRAM) at higher quality
motion-mirror run character.png motion.mp4 \
  --backend wan-14b-vace-gguf --frames 81 --steps 50 --resolution 480x832

# Let Motion Mirror pick from available VRAM
motion-mirror run character.png motion.mp4 --auto

# Gradio web UI
motion-mirror ui
```

Output → `./outputs/result.mp4`.

---

## CLI

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
| `--steps` | denoising steps, 1–200 (default 30; 50 for higher quality) |
| `--resolution` | `WxH`, e.g. `480x832` (portrait) or `832x480` |
| `--offload-model` / `--t5-cpu` | trade speed for VRAM |
| `--segmenter` | `rembg` (default) \| `sam2` |
| `--flow-estimator` | `farneback` (default) \| `raft` |

Presets: `default`, `low-vram`, `mock` — see `motion-mirror presets --list`.

---

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

---

## Backends

| Backend | VRAM (measured) | Identity | Notes |
|---|---:|---|---|
| `wan-1.3b-vace` | 8.0 GB | loose | Fast, lightest. Follows motion well; face/appearance can drift. |
| `wan-14b-vace-gguf` | ~18 GB | strong | **Recommended.** Q4_K_M quantized 14B — locks identity, faster than the full 14B. |
| `wan-14b-vace` | 8.0 GB* | strong | Full 14B. *Fits ~9 GB via sequential CPU offload, but slower. |
| `mock` | — | — | Solid-colour output for testing without a GPU. |

All backends are GPU-validated end-to-end (RTX 3090/4090/A6000). The 14B backends are the fix for the 1.3B model's loose identity adherence; `--auto` selects `wan-1.3b-vace` and never routes to a 14B backend implicitly.

---

## Known limitations

- **Single-person only** — multi-person reference videos raise `MultiplePeopleDetectedError`. Low-confidence background detections are filtered automatically; crop to one clear subject.
- **Background** — the output background follows the character image, not the reference video's scene.
- **Generation time** — under CPU offload a full 81-frame clip takes tens of minutes; much faster on 24 GB+ cards.
- **Drivers** — needs an NVIDIA driver new enough for your PyTorch CUDA build (570/CUDA 12.8 validated).

---

## Development

```bash
pip install -e ".[dev]"
pytest -m "not gpu"        # non-GPU suite (no weights/GPU needed) — runs in CI
pytest -m gpu             # GPU suite (needs CUDA + weights)
```

GPU validation is reproducible via [`runpod-validation/`](runpod-validation/README.md) (API-only RunPod harness, no SSH).

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Model weights and dependencies keep their own licenses; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Generated outputs are subject to the Wan2.1 model terms (Apache 2.0, commercial use permitted).

---

Repo: [github.com/halli75/motion-mirror](https://github.com/halli75/motion-mirror) · [Issues](https://github.com/halli75/motion-mirror/issues)
