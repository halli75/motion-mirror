# Motion Mirror — Session Handoff

Last updated: 2026-07-04 (v0.3: VACE-only backend shift). Repo: https://github.com/halli75/motion-mirror

## Project

Local-first motion transfer pipeline: animate a character image from a
reference motion video. v0.3 consolidates onto the one backend that GPU
validation proved works.

## THE HEADLINE

**v0.3 is VACE-only.** GPU validation (2026-07-03, RTX 3090/4090) showed
that of the five real backends, only `wan-1.3b-vace` delivers
trajectory-driven motion transfer (PASS end-to-end, 8.02 GB peak). The
wan-move family (14b/fast/gguf) discarded the dense trajectory (prompt-only
motion) and concat-id required an unmerged DiffSynth fork — all four were
deleted along with their presets, extras, docs, tests, and harness roles.

Surviving surface:
- Backends: `auto` / `wan-1.3b-vace` / `mock` (config default `wan-1.3b-vace`).
- `generate/controlnet.py` → `generate/vace.py` (`generate_with_vace`);
  "controlnet" alias dropped.
- `hardware.py` single tier: ≥9.02 GB free VRAM → vace with
  `offload_model` + `t5_cpu`, else `InsufficientVRAMError`.
- Presets: `default` / `low-vram` / `mock`. Version `0.3.0a0`.
- Kept: DWPose extraction, SAM-2 (segmenter + reference-masker), ComfyUI
  nodes, gpu-inference extra.
- New: 4k+1 frame validation on the real VACE path; try/finally CUDA cache
  cleanup around generation (review-wave find).

Known limitation: reference-image identity adherence is loose at 1.3B scale
(GPU run rendered a different-looking dancer than the reference; motion and
scene were correct). Scaling identity quality (e.g. VACE-14B) is future work.

## Current state

- Branch `runpod-v02a-validation`, PR #7 open → main. The v0.3 shift is
  committed on top of the Wave-1/2/2b validation work.
- Suite: **228 passed** (non-GPU), 7 gpu-marked deselected. Wheel builds
  clean (`motion_mirror-0.3.0a0`). shellcheck clean.
- Implementation: 5 parallel Opus packages (core src / interfaces / tests /
  docs / harness) + 5-reviewer Sonnet wave. Review found: comfyui README
  staleness, phantom `MOTION_MIRROR_MODEL_DIR` env var in windows-install,
  dead multi-source download machinery in cli.py, hand-duplicated
  valid_backends, missing CUDA-cleanup try/finally + tests, preset
  frame-rule guard — all fixed. False positives (kept as-is):
  `recommend_backend` tuple signature (spec'd), `vace` single-item download
  group (spec'd), `generate/__init__` re-export (public API), the
  `cfg.backend or request.backend` mock check (two legit entry points).
- **GPU validation of the v0.3 tree NOT yet run — awaiting user approval**
  (gpu-validate.yml is ready; single vace role, vace + sam2-vace smokes).
- RunPod: 0 pods, session spend ~$2.03, balance ~$5.1. Never count or
  terminate pods this orchestrator didn't create.

## Validation harness (runpod-validation/)

API-only, no SSH. Single role `vace` (3090/4090 spot, 140 GB disk, 3.5 h
cap, $4.50 spend guard). `orchestrate.py run` (role defaults to vace) +
`pod_bootstrap.sh` (groups dwpose/vace/extras; vace + sam2-vace smokes;
status/heartbeat via :8000 proxy). Evidence → `runpod-validation/evidence/`.
`RUNPOD_API_KEY` is set as a GitHub secret; gpu-validate.yml is
workflow_dispatch with no inputs.

## Next steps

1. **User-approved GPU confirm run** of the v0.3 tree (vace + sam2-vace,
   ~$0.15-0.30). Judge by eyes on frame strips first, structural gate
   second, then send video.
2. 81-frame follow-up (doc rule: 17-frame pass owes an 81-frame confirm).
3. Merge PR #7 once GPU-confirmed.
4. Future: identity quality (VACE-14B research: GGUF quant availability,
   newer Wan-base VACE checkpoints).

## Gotchas / lessons (carried forward)

- Judge GPU output by EYES on frame strips first — structural gate second
  (numbers lied twice; user's review caught identity/background/motion
  detail my strip review missed).
- Local background orchestrators get reaped — pod runs autonomously,
  Monitor polls proxy status.json, fetch+terminate on DONE.
- pytest here can silently run the INSTALLED wheel, not src/ — use
  `PYTHONPATH=src` (or editable install) for local runs.
- bash `GROUPS` is readonly — prefix pod-script vars MM_*.
- Content gates judge structure, not luminance.
- RunPod spend: per-pod costPerHr×uptime; account balance is polluted by
  other projects' pods.
- Pipeline rejects .webm/.ogv — pod transcodes with ffmpeg.
- gh works after `export GH_TOKEN=$(git credential fill ...)`.

## Key commands

```powershell
$env:PYTHONPATH="src"; pytest -m "not gpu" -q --tb=short   # 228 green
python runpod-validation/orchestrate.py preflight
python runpod-validation/orchestrate.py run                # role defaults to vace
```
