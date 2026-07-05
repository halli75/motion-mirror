# Motion Mirror — Session Handoff

Last updated: 2026-07-04 (v0.4.0a0: sam2-masker removal + 14B VACE backends, GPU-pending). Repo: https://github.com/halli75/motion-mirror

## Project

Local-first motion transfer: animate a character image from a reference motion
video. Validated core: `wan-1.3b-vace` (motion + scene correct, identity loose
at 1.3B scale).

## THE HEADLINE

**v0.4.0a0 is implemented and locally verified, GPU validation NOT yet run.**
Two workstreams landed on `runpod-v02a-validation`:

1. **sam2 reference-masker mode DELETED.** Two GPU runs proved it broken by
   design: VACE "keep" regions copy from the skeleton-on-black control video,
   so a subject stencil always yields subject-on-black-void. (The mask
   polarity bug found first was real and fixed, but the mode's ceiling made it
   pointless.) `segmenter="sam2"` (character-image segmentation) KEPT.
2. **Two 14B backends ADDED (GPU-UNVALIDATED):** `wan-14b-vace`
   (Wan-AI/Wan2.1-VACE-14B-diffusers, ~75 GB) and `wan-14b-vace-gguf`
   (QuantStack Q4_K_M transformer 11.6 GB + ~12 GB base via allow_patterns).
   Explicit `--backend` only — **auto never selects them** (no measured VRAM
   floor). Motivation: the 1.3B loose-identity limitation; 14B is the untested
   candidate fix.

## Credit-protection status (why the next GPU run should work first try)

- `scripts/verify_model_specs.py`: every download spec verified against the
  live HF API — **7/7 PASS** (repos, exact filenames, sizes ±15%,
  allow_patterns coverage). Wired into `orchestrate.py preflight`.
- It corrected 3 wrong sizes: wan-1.3b-vace is really ~19 GB (not 5),
  sam2 ~1.8 GB, 14B base ~12 GB (gguf backend total ~24 GB, not 55).
- GGUF loading path: diffusers gained WanVACETransformer3DModel single-file
  support in **0.35.0** → pinned `diffusers>=0.35.0` + `gguf>=0.10.0`;
  exact-args mocked tests lock the loader call; gguf uses
  `enable_model_cpu_offload()` (sequential offload corrupts GGUF quant
  metadata → KeyError: None). Residual risk: 14B GGUF inference untested
  upstream (diffusers #11878 hit the 1.3B variant); user declined the local
  11.6 GB CPU load-check.
- Harness prepared, NOT run: 3-smoke matrix (1.3B regression → gguf →
  full 14B; gguf deliberately BEFORE the full download so the base-cache
  resolver path is exercised), disk 200 GB.
  **TODO(gpu-run) decisions:** wall cap 3.5 h / spend cap $4.50 are too tight
  for ~118 GB downloads + two offloaded 14B smokes (propose ~6 h); heartbeat
  stall guard (15 min) may trip on silent 75 GB shard loads.

## Current state

- Branch `runpod-v02a-validation`, PR #7. Suite: **250 passed / 16 skipped
  (network-marked) / 7 gpu-deselected**. Wheel 0.4.0a0 builds + installs.
  shellcheck clean. `pytest -m network` (16 live HF tests) green.
- 4-reviewer wave ran (correctness/tests/docs/harness); all confirmed
  findings fixed: stale ImportError hint, gguf import-check, dtype follows
  device, pipeline-level conditioning test for 14B backends, docs headings,
  gguf license entry, LICENSE file added (was a broken README link),
  spec-verifier wired into preflight, vestigial MM_GROUPS removed.
- v0.3 history: VACE-only consolidation (1e6deb2), sam2 polarity fix
  (faae660), GPU runs 2026-07-04: vace PASS ×3 (8.02 GB), sam2-vace dead end.
- RunPod: 0 pods ours, balance ~$4.19. Never count/terminate pods this
  orchestrator didn't create.

## Next steps

1. **GPU validation run (user approval + cap decision needed):** raise wall
   cap to ~6 h, confirm spend (~$1.50-3.00 at 3090/4090 spot for downloads +
   3 smokes), then `orchestrate.py run`. Judge by eyes on frame strips first.
   Key question: does 14B fix identity?
2. Merge PR #7 once validated.
3. If 14B identity is good → consider promoting a 14B tier into auto with the
   measured floor; possibly retire 1.3B or keep as low-VRAM tier.

## Key commands

```powershell
$env:PYTHONPATH="src"; pytest -m "not gpu" -q      # 250 green
$env:PYTHONPATH="src"; pytest -m network -q        # live HF spec checks
python scripts/verify_model_specs.py               # 7/7 PASS expected
python runpod-validation/orchestrate.py preflight  # includes spec gate
```

## Gotchas (carried forward)

- Judge GPU output by EYES on frame strips first; structural gate second.
- pytest here can silently run the INSTALLED wheel — always PYTHONPATH=src.
- Python write_text on Windows writes CRLF — pod_bootstrap.sh must stay LF
  (shellcheck SC1017); use write_bytes or newline="\n".
- Local background orchestrators get reaped — pod runs autonomously; poll
  proxy status.json via Monitor; `attach --pod-id X` exists; terminate via
  `orchestrate.py terminate --pod-id X`.
- bash `GROUPS` readonly; RunPod balance polluted by other projects' pods;
  .webm inputs need transcode.
