# Motion Mirror — Session Handoff

Last updated: 2026-07-03 (Wave-2/2b: VACE works, CI hardened). Repo: https://github.com/halli75/motion-mirror

## Project

Local-first motion transfer pipeline: animate a character image from a
reference motion video. v0.2 goal was consumer-hardware tiers (8–24 GB).

## THE HEADLINE (read this first)

**VACE motion transfer now works end-to-end** (Wave-2b confirm run, zero
failures): `wan-1.3b-vace` renders a photorealistic dancer following the
extracted plié motion. Two fixes got it there: (1) canonical OpenPose-18
conditioning skeleton (COCO-17 + custom colors was out-of-distribution —
model echoed the control), (2) subject-centric prompt (the old prompt said
"skeleton motion"; text dominates subject choice, so it drew a skeleton).

Still true: the wan-move family (14b/fast/gguf) **discards the dense
trajectory** — prompt-text conditioning only (register #1, Wave 3, needs the
real `wan.WanMove` runtime). And VACE's reference-image identity adherence
is loose (male dancer vs referenced ballerina) — strong identity is the
Concat-ID track.

Backend truth (RTX 3090/4090, 17-frame smokes; evidence in
`runpod-validation/evidence/wave2*`):
- `wan-1.3b-vace` — **PASS**, 8.02 GB peak. `vace2_result.mp4`.
- `wan-move-gguf` — **PASS** (fp32-VAE path), 11.52 GB peak, renders the
  reference ballerina; motion prompt-only. `gguf2_result.mp4`.
- `wan-move-fast` — blocked: flash_attn ABI (Wave 3).
- `wan-1.3b-concat-id` — needs unmerged Concat-ID DiffSynth fork.
- `wan-move-14b` — never GPU-run (48 GB pod deferred).
- 12 GB `t5_cpu`-only tier — probed, **removed** (15.07 GB transient peak +
  bf16→fp32-VAE `slow_conv3d` crash without offload hooks).

## Current state

- Branch `runpod-v02a-validation`; **PR #7 open → main**: harness + Wave-1
  (25 register items) + Wave-2 (shellcheck CI, gpu-validate.yml
  workflow_dispatch button, OpenPose-18 skeleton, orchestrator gql retry)
  + Wave-2b (prompt fix, gguf first-time resolver, tier removal, measured
  docs). Suite: **265 passed**. 81-frame follow-ups still owed (doc rule).
- `RUNPOD_API_KEY` set as a GitHub secret; gpu-validate workflow usable
  after merge to main.
- RunPod: **0 pods running**, session spend ~$2.03, balance ~$5.1.
  Key: user provides; account may have OTHER projects' pods — never count
  or terminate pods this orchestrator didn't create (state file guards this).

## What Wave 1 fixed (in PR #7)

Crashes (pose zero-detection IndexError + nearest-centroid tracking,
concat-id/fast VRAM teardown) · harness trust (canonical result.mp4 for
silent clips; structural content gate — fill-ratio + edge-IoU, replacing
mean-luminance that failed BOTH directions on real output; work-derived pod
heartbeat) · hardware routing (floors = measured peak + _HEADROOM_GB;
wan-move download size 28→42 GB, _DISK_MARGIN) · model sources (resolvers
raise on incomplete cache instead of silent re-download; fp32 VAE for GGUF)
· trajectory numerics (per-frame confidence hold-last, isotropic bbox-scaled
Gaussian, true grid fallback, alpha>127) · hygiene (config enum validation,
UI 4k+1 frame snap, preset parity).

Earlier in the session (same branch): meta-tensor t5_cpu+offload crash fix,
GGUF enable_model_cpu_offload, VACE all-white mask (necessary, insufficient).

## Next steps

1. **Merge PR #7** (CI green; CodeRabbit review optional).
2. **81-frame follow-up run** for vace + gguf (doc rule: 17-frame pass owes
   an 81-frame confirm; expect ~4-6x generation time).
3. **Wave 3** (architecture): real trajectory conditioning via wan.WanMove
   runtime (#1), Concat-ID fork for identity (#3) — also the fix for VACE's
   loose identity adherence — flash_attn wheel/image for fast (#4),
   wan-move-14b 48 GB run.

## Validation harness (runpod-validation/)

API-only, no SSH. `orchestrate.py` (preflight / run --role a|b / attach /
terminate; spend guard scoped to OUR pods only, $4.50 cap; MM_TIER_A=1 env
→ lean vace/gguf/fast run) + `pod_bootstrap.sh` (phased, status/heartbeat
served via http.server :8000 through the RunPod proxy) + `validate_inputs.py`
(pipeline-exact DWPose Wholebody gate for smoke inputs; samples.json =
Plié ballet tutorial clip [4,10]s + Matosinhos ballet dancer CC0 image).
Evidence lands in runpod-validation/evidence/ (untracked this session).

## Gotchas / lessons (this session)

- **Long-running local background processes get reaped** (~3 orchestrator
  deaths). Pattern that works: pod runs autonomously; a Monitor polls the
  proxy status.json; fetch+terminate manually on DONE. `attach` subcommand
  exists for re-entry.
- **bash `GROUPS` is a special readonly array** — assignments silently
  ignored. Prefix pod-script vars MM_*.
- **flash-attn pip install without --no-deps clobbers the image's CUDA
  torch** (its torch dep reinstalls CPU torch). Always --no-deps + verify
  `import flash_attn` + re-verify torch.cuda after.
- **Content gates must judge structure, not luminance** — mean-pixel checks
  passed a skeleton-on-blue and failed a dancer-on-black. Frame-strip visual
  review caught what the numbers lied about, twice.
- **RunPod spend guard**: use per-pod costPerHr×uptime with ended_at
  recorded on terminate; account balance deltas are polluted by other
  projects' pods; `currentSpendPerHr` lags after termination — trust the
  pod list.
- Pipeline rejects .webm/.ogv (mp4/mov/avi/mkv only) — pod transcodes with
  ffmpeg (libx264 → mpeg4 fallback).
- extract_pose person-count gate is **frame-0 only** and uses raw Wholebody
  detector count — validate smoke inputs with the pipeline's own stack
  (validate_inputs.py does).
- gh works after `export GH_TOKEN=$(git credential fill ...)` (see history).

## Key commands

```powershell
pytest -m "not gpu" -q --tb=short          # 260 green
python runpod-validation/orchestrate.py preflight
$env:MM_TIER_A="1"; python runpod-validation/orchestrate.py run --role a
```
