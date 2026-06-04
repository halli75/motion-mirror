# Windows Installation Guide

Windows is supported with caveats. The most common failures are mismatched
PyTorch CUDA wheels, old NVIDIA drivers, missing Microsoft Visual C++ runtime
libraries, and model caches on drives without enough free space.

## Prerequisites

- Windows 10 or 11
- Python 3.11
- Recent NVIDIA driver compatible with CUDA 12.x
- Microsoft Visual C++ Redistributable 2015-2022
- 50-80 GB free disk space for model weights

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install the CUDA build first. Pick the CUDA index that matches your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Motion Mirror from a checkout.
pip install -e ".[cuda,gpu-inference]"
```

Optional v0.2a runtimes:

```powershell
# LightX2V fast backend
pip install -e ".[lightx2v]"

# Experimental GGUF backend
pip install -e ".[gguf]"

# SAM-2 segmenter and reference-video masker
pip install git+https://github.com/facebookresearch/sam2.git
```

## Verify The Install

```powershell
python -c "import motion_mirror; print('motion_mirror OK')"
python -c "from motion_mirror.cli import app; print('cli OK')"
python -c "from motion_mirror.ui.app import create_app; print('ui OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
motion-mirror --help
motion-mirror presets --list
```

If `torch.cuda.is_available()` is `False`, real generation will not use the GPU.
Fix the CUDA/PyTorch/driver mismatch before downloading large models.

## Model Cache

Downloads go to `~/.cache/motion-mirror` by default. To use another drive:

```powershell
$env:MOTION_MIRROR_MODEL_DIR = "D:\motion-mirror-models"
motion-mirror download --model dwpose
```

Run downloads explicitly before long generation jobs so disk-space failures
happen early.

## Smoke Test

CPU-only mock smoke:

```powershell
motion-mirror run character.png motion.mp4 --backend mock --frames 3 --resolution 64x32
```

CUDA backend smoke after weights are downloaded:

```powershell
$env:PYTHONPATH = "C:\Users\arnav\motion-mirror\src"
python scripts\v02a_gpu_smoke.py `
  --image character.png `
  --motion motion.mp4 `
  --backend wan-1.3b-vace `
  --report outputs\v02a-smoke\vace.json
```

## Troubleshooting

- `torch.cuda.is_available() == False`: install the correct CUDA wheel and update
  the NVIDIA driver.
- `ImportError` for optional backends: install the matching optional extra listed
  above.
- Disk-space errors: set `MOTION_MIRROR_MODEL_DIR` to a drive with enough space.
- Long Gradio jobs behind a proxy may disconnect. Use the CLI for the most
  reliable long generation path.
