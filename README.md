# Motion Mirror

**Local-first motion transfer — animate any character image from a reference video.**

Motion Mirror is the open-source alternative to Kling AI's Motion Control. Give it a character image and a reference video; it produces an animated video of your character performing the same motion. Everything runs on your machine — no cloud, no API keys, no per-clip fees.

> **Early release** — v0.3 ships a focused, GPU-validated backend lineup built around the Wan2.1-VACE 1.3B model. See [Known Limitations](#known-limitations) before installing.

---

## How it works

```
character.png + motion_video.mp4
        │
        ▼
 [1] Segment character     rembg or SAM-2 removes background -> RGBA mask
        │
        ▼
 [2] Extract pose          DWPose-L detects skeleton keypoints per frame
        │
        ▼
 [3] Build conditioning    Skeleton + mask conditioning frames for VACE
        │
        ▼
 [4] Generate video        wan-1.3b-vace (Wan2.1-VACE 1.3B) or mock
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
| GPU VRAM | ~9 GB free for `wan-1.3b-vace` | 12 GB+ free |
| System RAM | 32 GB | 64 GB |
| Disk space | 30 GB free | 50 GB free |
| CUDA | 12.x | 12.x |
| Python | 3.11 | 3.11+ |

CPU-only mode is not supported for real generation (mock mode works for testing).

> **VRAM note:** `wan-1.3b-vace` measured an **8.02 GB peak** on the RTX 3090/4090 validation runs (2026-07-03, 17-frame smoke, `--offload-model --t5-cpu`). Auto-selection requires **≥9.02 GB free VRAM** (measured peak + 1 GB headroom); below that floor, `--auto` cannot run real generation and you should use `--backend mock`.

---

## Installation

### 1. Install PyTorch with CUDA first

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Motion Mirror with GPU inference dependencies

```bash
pip install -e ".[cuda,gpu-inference]"
```

Or from PyPI once published:

```bash
pip install "motion-mirror[cuda,gpu-inference]"
```

### 3. Optional: SAM-2 segmenter / reference-video masker

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 4. Download model weights

```bash
# Wan2.1-VACE-1.3B generation model (~5 GB, diffusers format)
motion-mirror download --model wan-1.3b-vace

# DWPose-L pose estimation (detector + pose, ~350 MB)
motion-mirror download --model dwpose

# SAM-2 segmenter / reference-video masker (~900 MB)
motion-mirror download --model sam2
```

`--model` also accepts groups: `vace`, `dwpose`, `extras`, or `all`. Downloads go
to `~/.cache/motion-mirror/`. A disk-space check runs before each download.

---

## Quick start

```bash
# Basic run (defaults to wan-1.3b-vace with offload)
motion-mirror run character.png motion.mp4

# Low-VRAM VACE path made explicit
motion-mirror run character.png motion.mp4 \
  --backend wan-1.3b-vace \
  --offload-model \
  --t5-cpu

# Let Motion Mirror pick from available CUDA VRAM
motion-mirror run character.png motion.mp4 --auto

# Explicit options with SAM-2 segmentation
motion-mirror run character.png motion.mp4 \
  --backend wan-1.3b-vace \
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

### Public run options

```bash
motion-mirror run character.png motion.mp4 \
  --backend auto|wan-1.3b-vace|mock \
  --auto \
  --offload-model \
  --t5-cpu \
  --flow-estimator raft \
  --segmenter sam2 \
  --reference-masker sam2
```

`--reference-masker sam2` (SAM-2 propagation over the reference video) is the
least battle-tested option; segmentation of the character image with
`--segmenter sam2` is validated. Non-GPU CI covers config, CLI, routing, and
mocked backend contracts.

> **Conditioning note:** `wan-1.3b-vace` conditions the Wan2.1-VACE pipeline on
> per-frame skeleton (OpenPose-18) and mask frames derived from the reference
> video. Reference-image identity adherence is loose at 1.3B scale — see
> [Known Limitations](#known-limitations).

### Presets

```bash
motion-mirror presets --list
```

| Preset | Resolution | Frames | Density | Notes |
|---|---|---|---|---|
| `default` | 832×480 | 81 | 512 | Standard quality, `wan-1.3b-vace` |
| `low-vram` | 832×480 | 81 | 512 | `wan-1.3b-vace` with offload / T5 CPU |
| `mock` | 128×64 | 4 | 32 | For testing without GPU |

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
    MultiplePeopleDetectedError,  # >1 person detected (single-person only)
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

**Loose identity adherence at 1.3B scale**
The Wan2.1-VACE 1.3B backend follows the reference motion well but does not
strongly lock onto the input character's face and appearance — identity can
drift, especially during large head movements or fast motion. This is a known
limitation of the 1.3B model, not a bug. Strong identity preservation is a
larger-model research problem and is not addressed at this scale.

**Single-person only**
Multi-person reference videos raise `MultiplePeopleDetectedError`. Crop the
reference video to one person before running Motion Mirror.

**SAM-2 reference-video masking is experimental**
`--segmenter sam2` (character-image segmentation) is validated. `--reference-masker sam2`
(SAM-2 mask propagation across the reference video) is newer and less tested.

**~9 GB+ free VRAM required for real generation**
The 1.3B VACE backend measured an 8.02 GB peak with `--offload-model --t5-cpu`
and needs ~9 GB free VRAM to run. CPU offloading uses system RAM as overflow
storage during inference. Cards below this floor should use `--backend mock`.

**~5 GB model download**
First run requires downloading ~5 GB (Wan2.1-VACE-1.3B) + ~350 MB (DWPose),
plus ~900 MB if SAM-2 is used. Allow ~30 GB free disk space for caches and
working files.

**Generation time**
With sequential CPU offloading (required at ~9–12 GB VRAM), a short 17-frame
clip takes several minutes; an 81-frame clip (~5 seconds at 16 fps) takes
substantially longer. Generation is faster on larger cards (A100/H100) with more
VRAM headroom.

---

## Validation Status

Non-GPU CI passes for config, CLI, routing, mocks, and package imports. GPU
validation was run on RunPod (RTX 3090/4090, 2026-07-03) via `runpod-validation/`:

| Backend | Target Hardware | Status |
|---|---:|---|
| `wan-1.3b-vace` | ~9–12 GB free VRAM | **PASS** — end-to-end motion transfer, 8.02 GB peak (17-frame smoke) |
| `mock` | CPU / any | Testing only, no real generation |

See [`docs/windows-install.md`](docs/windows-install.md) for Windows setup notes.

---

## Roadmap

| Version | Focus | Key additions |
|---|---|---|
| **v0.1** | End-to-end pipeline | Trajectory synthesis, CLI, UI |
| **v0.2** | Hardware accessibility | 1.3B VACE backend, RAFT / SAM-2 options |
| **v0.3** *(current)* | VACE-only lineup | GPU-validated `wan-1.3b-vace`, ComfyUI nodes |
| **v0.4** | Quality | Face restoration, frame interpolation, CI benchmarks |
| **v0.5** | Community | LoRA fine-tuning, batch mode, docs site |
| **v1.0** | Stable | Preset library, PyPI + Docker, stable Python API |

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
</content>
</invoke>
