# v0.2a Hardware Validation

Motion Mirror v0.2a is not release-ready until the public hardware table is
backed by measured runs. Non-GPU CI proves routing and contracts; it does not
prove VRAM accessibility or output quality.

## Required Evidence

For each backend, save:

- `scripts/v02a_gpu_smoke.py` JSON report
- full terminal log
- generated MP4
- GPU model and VRAM
- peak CUDA memory
- wall-clock runtime
- whether OpenCV can read the output MP4

If a backend fails or exceeds its advertised tier, update the README and
hardware auto-selection policy before release.

## Smoke Inputs

Use one short, known-good character image and one short single-person reference
video. Start with `--frames 17` and `--density 256` to keep validation cost low.
Run an 81-frame follow-up only after the 17-frame smoke succeeds.

## Commands

```powershell
$env:PYTHONPATH = "C:\Users\arnav\motion-mirror\src"

python scripts\v02a_gpu_smoke.py `
  --image C:\path\to\character.png `
  --motion C:\path\to\motion.mp4 `
  --backend wan-1.3b-vace `
  --output-dir outputs\v02a-smoke\vace `
  --report outputs\v02a-smoke\vace.json

python scripts\v02a_gpu_smoke.py `
  --image C:\path\to\character.png `
  --motion C:\path\to\motion.mp4 `
  --backend wan-move-fast `
  --output-dir outputs\v02a-smoke\fast `
  --report outputs\v02a-smoke\fast.json

python scripts\v02a_gpu_smoke.py `
  --image C:\path\to\character.png `
  --motion C:\path\to\motion.mp4 `
  --backend wan-move-gguf `
  --output-dir outputs\v02a-smoke\gguf `
  --report outputs\v02a-smoke\gguf.json

python scripts\v02a_gpu_smoke.py `
  --image C:\path\to\character.png `
  --motion C:\path\to\motion.mp4 `
  --backend wan-move-14b `
  --output-dir outputs\v02a-smoke\full `
  --report outputs\v02a-smoke\full.json

python scripts\v02a_gpu_smoke.py `
  --image C:\path\to\character.png `
  --motion C:\path\to\motion.mp4 `
  --backend wan-1.3b-vace `
  --reference-masker sam2 `
  --output-dir outputs\v02a-smoke\sam2-vace `
  --report outputs\v02a-smoke\sam2-vace.json
```

## Hardware Matrix

| Backend | Target Tier | Required Result |
|---|---:|---|
| `wan-1.3b-vace` | 8-12 GB | MP4 readable, peak VRAM within tier, VACE conditioning accepted |
| `wan-move-fast` | 24 GB | LightX2V imports, runtime config accepted, MP4 readable |
| `wan-move-gguf` | 12-16 GB | GGUF transformer loads, offload works, MP4 readable |
| `wan-move-14b` | 40 GB+ | Full backend remains readable and stable |
| `wan-1.3b-vace` + `--reference-masker sam2` | 24 GB+ validation box | SAM-2 propagated mask video is produced and consumed |

## Validation Status (RunPod, RTX 3090/4090, 2026-07-03)

Measured on 17-frame smokes via `runpod-validation/` (81-frame follow-ups
still owed per the rule above):

| Backend | Result | Peak VRAM |
|---|---|---:|
| `wan-1.3b-vace` | **PASS** — photorealistic dancer following the extracted motion (canonical OpenPose-18 conditioning + subject-centric prompt). Reference-image identity adherence is loose; strong identity is the Concat-ID track. | 8.02 GB |
| `wan-move-gguf` | **PASS** — renders the reference ballerina (fp32-VAE path). Motion is prompt-only until Wave-3 trajectory conditioning. | 11.52 GB |
| `wan-move-fast` | FAIL — LightX2V hard-imports `flash_attn`; prebuilt wheels fail ABI import on the pod image. Needs a custom wheel/image (Wave 3). | — |
| `wan-1.3b-concat-id` | Not run — requires the unmerged Concat-ID DiffSynth fork. | — |
| `wan-move-14b` | Not run — 48 GB pod deferred. | — |

The former 12 GB `t5_cpu`-only middle tier was probed and removed:
`pipe.to(cuda)` transiently peaked at 15.07 GB (T5 lands on GPU before the
`t5_cpu` move) and generation crashed at VAE encode (bf16 input into the
fp32 VAE without accelerate's auto-casting hooks). 12 GB cards use the
fully-offloaded vace tier.
