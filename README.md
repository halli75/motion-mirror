# Motion Mirror

**Local-first motion transfer — animate any character image from a reference video.**

Motion Mirror is the open-source alternative to Kling AI's Motion Control. Give it a character image and a reference video; it produces an animated video of your character performing the same motion. Everything runs on your machine — no cloud, no API keys, no per-clip fees.

> **Early release** — v0.2a expands Motion Mirror toward consumer hardware with low-VRAM, fast, and experimental GGUF backends. See [Known Limitations](#known-limitations) before installing.

---

## How it works

```
character.png + motion_video.mp4
        │
        ▼
 [1] Segment character     rembg or SAM-2 removes background -> RGBA mask
        │
        ▼
 [2] Extract pose          DWPose-L detects 133 skeleton keypoints per frame
        │
        ▼
 [3] Synthesize trajectory 3-layer dense point tracks:
                           Layer 1 — skeleton anchors
                           Layer 2 — Gaussian-falloff interpolation
                           Layer 3 — optical flow (Farneback or RAFT)
        │
        ▼
 [4] Generate video        wan-move-14b, wan-move-fast,
                           wan-move-gguf, wan-1.3b-vace, or mock
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
| GPU VRAM | 8 GB for `wan-1.3b-vace` | 24 GB+ for fast/full backends |
| System RAM | 32 GB | 64 GB |
| Disk space | 50 GB free | 80 GB free |
| CUDA | 12.x | 12.x |
| Python | 3.11 | 3.11+ |

CPU-only mode is not supported for real generation (mock mode works for testing).

> **VRAM note:** `--auto` chooses from free CUDA VRAM: 8-12 GB uses `wan-1.3b-vace`, 16-24 GB uses experimental `wan-move-gguf`, 24-40 GB uses `wan-move-fast`, and 40 GB+ uses `wan-move-14b`.

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

Optional v0.2a runtimes:

```bash
# LightX2V fast backend
pip install -e ".[lightx2v]"

# Experimental GGUF backend
pip install -e ".[gguf]"

# SAM-2 segmenter / reference-video masker
pip install git+https://github.com/facebookresearch/sam2.git
```

### 4. Download model weights

```bash
# Wan2.1-I2V-14B generation model (~28 GB, diffusers format)
motion-mirror download --model wan-move

# Experimental Wan2.1-I2V-14B GGUF transformer (~12 GB)
motion-mirror download --model gguf

# Wan2.1-VACE-1.3B low-VRAM backend
motion-mirror download --model wan-1.3b-vace

# LightX2V 4-step fast backend assets
motion-mirror download --model fast

# DWPose-L pose estimation (~350 MB)
motion-mirror download --model dwpose

# SAM-2 segmenter / reference-video masker
motion-mirror download --model sam2
```

Downloads go to `~/.cache/motion-mirror/`. A disk-space check runs before each download.

---

## Quick start

```bash
# Basic run
motion-mirror run character.png motion.mp4

# High quality (1280×720, density 1024)
motion-mirror run character.png motion.mp4 --preset hq

# Let Motion Mirror pick from available CUDA VRAM
motion-mirror run character.png motion.mp4 --auto

# Low-VRAM v0.2a path
motion-mirror run character.png motion.mp4 \
  --backend wan-1.3b-vace \
  --offload-model \
  --t5-cpu

# Explicit options
motion-mirror run character.png motion.mp4 \
  --backend wan-move-gguf \
  --resolution 832x480 \
  --frames 81 \
  --density 512 \
  --flow-estimator raft \
  --segmenter sam2 \
  --reference-masker sam2 \
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

### v0.2a public options

```bash
motion-mirror run character.png motion.mp4 \
  --backend wan-move-14b|wan-move-fast|wan-move-gguf|wan-1.3b-vace|mock|auto \
  --auto \
  --offload-model \
  --t5-cpu \
  --flow-estimator raft \
  --segmenter sam2 \
  --reference-masker sam2
```

`wan-move-gguf` and `--reference-masker sam2` are experimental until real GPU validation is complete. Non-GPU CI covers their config, CLI, routing, and mocked backend contracts.

### Presets

```bash
motion-mirror presets --list
```

| Preset | Resolution | Frames | Density | Notes |
|---|---|---|---|---|
| `default` | 832×480 | 81 | 512 | Standard quality |
| `hq` | 1280×720 | 81 | 1024 | Higher quality, more VRAM |
| `mock` | 64×32 | 3 | 16 | For testing without GPU |
| `low-vram` | 832×480 | 81 | 512 | `wan-1.3b-vace` with offload/T5 CPU |
| `fast` | 832×480 | 81 | 512 | true LightX2V 4-step backend |
| `gguf` | 832×480 | 81 | 512 | experimental GGUF-quantized Wan backend |

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
    backend="wan-1.3b-vace",
    resolution="832x480",
    num_frames=81,
    trajectory_density=512,
    offload_model=True,
    t5_cpu=True,
    flow_estimator="raft",
    segmenter="sam2",
    reference_masker="sam2",
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

**Identity drift (v0.2a — Wan backends)**
The character's face in the output may not closely match the input photo, especially during large head movements or fast motion. This is a fundamental property of the Wan2.1-I2V-14B model, which has no explicit face-identity conditioning. Identity preservation is tracked for v0.3 via reward-guided optimization (IPRO).

**Single-person only**
Multi-person reference videos raise `MultiplePeopleDetectedError`. Crop to one person, or use `--person-index N` (planned for v0.2).

**Backend maturity varies**
`wan-1.3b-vace`, `wan-move-fast`, `wan-move-gguf`, RAFT, and SAM-2 options are v0.2a accessibility features. `wan-move-gguf` and SAM-2 reference-video propagation are experimental until real GPU validation is complete.

**8 GB+ GPU required for real generation**
The 1.3B VACE backend targets 8-12 GB GPUs. The 14B full backend still requires much larger cards. CPU offloading uses system RAM as overflow storage during inference.

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
| **v0.1** | End-to-end pipeline | 14B backend, trajectory synthesis, CLI, UI |
| **v0.2a** *(current)* | Hardware accessibility | 1.3B VACE, LightX2V 4-step, GGUF backend, RAFT/SAM-2 options |
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

CI runs `pytest -m "not gpu"` on every push via GitHub Actions. GPU tests are separate/manual because they require CUDA hardware and model weights.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

Model weights and third-party dependencies retain their own licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full breakdown. Generated outputs are subject to the Wan2.1 model card terms (also Apache 2.0, commercial use permitted).
