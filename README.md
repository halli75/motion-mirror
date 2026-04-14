# Motion Mirror

**Local-first motion transfer — animate any character image from a reference video.**

Motion Mirror is the open-source alternative to Kling AI's Motion Control. Give it a character image and a reference video; it produces an animated video of your character performing the same motion. Everything runs on your machine — no cloud, no API keys, no per-clip fees.

> **Early release** — v0.1 produces real results but has known limitations. See [Known Limitations](#known-limitations) before installing.

---

## How it works

```
character.png + motion_video.mp4
        │
        ▼
 [1] Segment character     rembg removes background → RGBA mask
        │
        ▼
 [2] Extract pose          DWPose-L detects 133 skeleton keypoints per frame
        │
        ▼
 [3] Synthesize trajectory 3-layer dense point tracks:
                           Layer 1 — skeleton anchors
                           Layer 2 — Gaussian-falloff interpolation
                           Layer 3 — optical flow (non-rigid: hair, clothing)
        │
        ▼
 [4] Generate video        Wan2.1-I2V-14B via diffusers
        │
        ▼
 [5] Passthrough audio     Original audio muxed into output
        │
        ▼
   output.mp4
```

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 24 GB (RTX 3090 / 4090) | 32 GB+ (RTX 5090, A100) |
| System RAM | 32 GB | 64 GB |
| Disk space | 50 GB free | 80 GB free |
| CUDA | 12.x | 12.x |
| Python | 3.11 | 3.11+ |

CPU-only mode is not supported for real generation (mock mode works for testing).

> **VRAM note:** The 14B model uses sequential layer-by-layer CPU offloading, so it runs on 24–32 GB VRAM cards. System RAM is used as overflow storage during inference.

---

## Installation

### 1. Install PyTorch with CUDA first

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Motion Mirror

```bash
pip install -e ".[cuda]"
```

Or from PyPI once published:

```bash
pip install motion-mirror[cuda]
```

### 3. Install GPU inference dependencies

```bash
pip install "diffusers>=0.33" transformers accelerate ftfy
```

### 4. Download model weights (~28 GB total)

```bash
# Wan2.1-I2V-14B generation model (~28 GB, diffusers format)
motion-mirror download --model wan-move

# DWPose-L pose estimation (~350 MB)
motion-mirror download --model dwpose
```

Downloads go to `~/.cache/motion-mirror/`. A disk-space check runs before each download.

---

## Quick start

```bash
# Basic run
motion-mirror run character.png motion.mp4

# High quality (1280×720, density 1024)
motion-mirror run character.png motion.mp4 --preset hq

# Explicit options
motion-mirror run character.png motion.mp4 \
  --backend wan-move-14b \
  --resolution 832x480 \
  --frames 81 \
  --density 512 \
  --device cuda \
  --output-dir ./my_outputs

# Launch the Gradio web UI
motion-mirror ui
```

Output is written to `./outputs/result.mp4` by default.

---

## CLI reference

```
motion-mirror --help

Commands:
  run        Run the full motion transfer pipeline
  download   Download model weights to local cache
  presets    List available generation presets
  benchmark  Print system and GPU diagnostics
  ui         Launch the Gradio web UI
```

### Presets

```bash
motion-mirror presets --list
```

| Preset | Resolution | Frames | Density | Notes |
|---|---|---|---|---|
| `default` | 832×480 | 81 | 512 | Standard quality |
| `hq` | 1280×720 | 81 | 1024 | Higher quality, more VRAM |
| `mock` | 64×32 | 3 | 16 | For testing without GPU |

### Benchmark

```bash
motion-mirror benchmark           # Python + platform info
motion-mirror benchmark --gpu-info  # GPU name and VRAM
```

---

## Python API

```python
from pathlib import Path
from motion_mirror import MotionMirrorPipeline, MotionMirrorConfig

cfg = MotionMirrorConfig(
    backend="wan-move-14b",
    resolution="832x480",
    num_frames=81,
    trajectory_density=512,
    device="cuda",
)

pipeline = MotionMirrorPipeline(cfg)
result = pipeline.run(
    image_path=Path("character.png"),
    motion_video_path=Path("motion.mp4"),
)

print(result.output_path)        # Path to result.mp4
print(result.segmentation_path)  # RGBA PNG
print(result.trajectory_path)    # .npz trajectory map
```

### Exception types

```python
from motion_mirror import (
    NoPoseDetectedError,          # no person in reference video
    MultiplePeopleDetectedError,  # >1 person detected (v0.1: single-person only)
    SmallSubjectError,            # person occupies <5% of frame
    SmallSubjectWarning,          # person occupies 5–10% (warning, not error)
    UnsupportedImageError,        # unsupported image format
    UnsupportedVideoError,        # unsupported video format
    VideoDecodeError,             # video cannot be decoded
    MultipleCharactersError,      # >1 person in character image
)
```

All exceptions inherit from `MotionMirrorError`.

---

## Known Limitations

**Identity drift (v0.1 — 14B backend)**
The character's face in the output may not closely match the input photo, especially during large head movements or fast motion. This is a fundamental property of the Wan2.1-I2V-14B model, which has no explicit face-identity conditioning. Identity preservation is tracked for v0.3 via reward-guided optimization (IPRO).

**Single-person only**
Multi-person reference videos raise `MultiplePeopleDetectedError`. Crop to one person, or use `--person-index N` (planned for v0.2).

**24 GB+ GPU required**
The 14B model requires at minimum an RTX 3090 or 4090. Sequential CPU offloading is used automatically — system RAM absorbs model layers not currently in use, so 32 GB+ system RAM is recommended alongside 24 GB+ VRAM.

**~28 GB model download**
First run requires downloading ~28 GB (Wan2.1-I2V) + ~350 MB (DWPose). A fast internet connection and ~50 GB free disk space are needed.

**Generation time**
With sequential CPU offloading (required for ≤32 GB VRAM), a 17-frame clip takes ~8–10 minutes on an RTX 5090. An 81-frame clip (~5 seconds at 16 fps) takes approximately 40–50 minutes. Generation is substantially faster on A100/H100 with full VRAM capacity.

---

## Validated hardware

End-to-end GPU tests (6/6 passing) verified on:

| GPU | VRAM | Result |
|---|---|---|
| RTX 5090 (RunPod) | 32 GB | All tests pass ✅ |

---

## Roadmap

| Version | Focus | Key additions |
|---|---|---|
| **v0.1** *(current)* | End-to-end pipeline | 14B backend, trajectory synthesis, CLI, UI |
| **v0.2a** | Hardware accessibility | 1.3B + ControlNet (8–12 GB), LightX2V 4-step, GGUF quantization |
| **v0.2b** | Identity + ecosystem | Concat-ID (1.3B), ComfyUI nodes |
| **v0.3** | Quality | IPRO 14B identity, CodeFormer, RIFE interpolation, CI benchmarks |
| **v0.4** | Community | LoRA fine-tuning, batch mode, docs site |
| **v1.0** | Stable | 50+ presets, PyPI + Docker, stable Python API |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all non-GPU tests (no GPU or model weights needed)
pytest -m "not gpu" -v

# Run GPU tests (requires downloaded weights + CUDA GPU)
pytest -m gpu -v
```

CI runs `pytest -m "not gpu"` on every push via GitHub Actions (128 tests, no GPU required).

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

Model weights and third-party dependencies retain their own licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full breakdown. Generated outputs are subject to the Wan2.1 model card terms (also Apache 2.0, commercial use permitted).
