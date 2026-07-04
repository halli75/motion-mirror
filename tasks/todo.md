# v0.3 — Shift to VACE-only backend

Plan: delete dead backends (wan-move-14b/fast/gguf, concat-id), rename controlnet.py → vace.py,
single-tier hardware, clean presets/CLI/docs/CI/harness. Verify via gates + Sonnet review wave.
GPU testing deferred pending user approval.

## Wave 1 — parallel packages (Opus)
- [x] A core src: generate/ rename+deletions, models.py, config.py, hardware.py, pipeline.py, types.py
- [x] B interfaces: cli.py, ui/app.py, presets ×2, comfyui_nodes/, pyproject.toml
- [x] D docs: README, delete 5 stale docs, v02b-scope.md, windows-install.md, THIRD_PARTY_LICENSES
- [x] E validation infra: gpu-validate.yml, orchestrate.py, pod_bootstrap.sh, runpod-validation/README, v02a_gpu_smoke.py

## Wave 2 — tests (after A+B)
- [x] C tests: delete dead-backend tests, rewrite test_hardware.py (11 single-tier), rename controlnet→vace, 4 new tests

## Wave 3 — verification gates
- [x] Import checks (generate_with_vace, pipeline, cli, ui)
- [x] pytest -m "not gpu" all green (228 passed)
- [x] Grep sweep: wan-move|wan_move|concat.id|lightx2v|gguf|controlnet (survivors: render_skeleton controlnet_aux provenance + one intentional "dropped" note in v02b-scope.md)
- [x] diff -r presets/ src/motion_mirror/presets/ parity (only .gitkeep, pre-existing)
- [x] python -m build --wheel (motion_mirror-0.3.0a0)
- [x] shellcheck runpod-validation/*.sh
- [x] Sonnet 5 review wave (5 reviewers); triaged; real findings fixed; re-ran gates
- [x] Rewrite handoff.md; complete this checklist + review section; commit(s) on runpod-v02a-validation
- [x] STOP — report to user for live GPU testing approval

## Review

**Scope executed**: 4 dead backends deleted end-to-end (code, tests, presets, extras, docs,
CI, harness). controlnet.py → vace.py with history. Net diff ≈ +600/−3700 lines.

**Review wave results** (5 Sonnet reviewers over staged diff):
- correctness/dead-refs: clean, zero findings (mock-path 4k+1 risk traced safe).
- CI/harness: clean, zero findings.
- docs: comfyui README v0.2b/CodeFormer staleness (fixed); phantom MOTION_MIRROR_MODEL_DIR
  env var in windows-install (fixed — junction guidance instead); Pillow/ftfy license
  entries (fixed); v02b-scope.md filename cosmetic (deferred).
- tests: new tests mutation-verified non-tautological; REAL src bug — no try/finally around
  pipe() so CUDA cache cleanup skipped on exception (fixed + 2 tests); preset 4k+1 CI guard
  (added); comfyui dropdown test (added).
- elegance: dead multi-source download machinery in cli.py removed (~30 lines); valid_backends
  now derived from BackendName; stale ~8 GB comment fixed. False positives: recommend_backend
  signature (plan-spec'd), vace single-item group (plan-spec'd), generate/__init__ re-export
  (public API), request.backend mock check (two legit entry points).

**Final verification**: 228 passed / 7 gpu-deselected; wheel 0.3.0a0 builds + installs
(--help / presets --list / download --help smoke pass); shellcheck clean; grep sweep clean.

**Deferred**: GPU confirm run of v0.3 tree (user approval gate); 81-frame follow-up;
docs/v02b-scope.md rename; identity-quality research (VACE-14B).
