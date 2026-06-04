# Motion Mirror CI Failure

## Checklist
- [x] Fetch latest failing CI metadata/log evidence and identify the failed step.
- [x] Reproduce the likely traceback locally under a no-`torch` condition.
- [x] Patch the minimal failure surface.
- [x] Run import checks and non-GPU tests.
- [x] Record review/results.

## Evidence
- GitHub Actions run `24632146935` failed in job `Lint + non-GPU tests`.
- Public job metadata shows `Check imports` passed and `Run non-GPU tests` failed.
- Raw Actions logs require authenticated access that is not exposed in this session; no GitHub MCP resources are configured and `gh` is not installed.
- The public annotations API only reports `Process completed with exit code 1`.
- Local CI-style pytest passes on this machine because `torch` is installed.
- Simulating CI without `torch` reproduces the suspected failure:
  `tests/test_segment_sam2.py::test_segment_sam2_dispatch_calls_predictor` fails at `src/motion_mirror/extract/segment.py` when `_segment_sam2()` imports `torch`.

## Review
- `_segment_sam2()` now imports `torch` only for CUDA inference contexts.
- CPU and mocked SAM-2 paths use no-op contexts, so the non-GPU test contract does not require `torch`.
- The existing SAM-2 dispatch test now blocks `torch` through `sys.modules` to cover the CI dependency shape.

## Verification
- `python -c "import motion_mirror; print('motion_mirror OK')"`: passed.
- `python -c "from motion_mirror.cli import app; print('cli OK')"`: passed.
- `python -c "from motion_mirror.ui.app import create_app; print('ui OK')"`: passed.
- No-`torch` targeted reproduction: `tests/test_segment_sam2.py::test_segment_sam2_dispatch_calls_predictor` passed.
- `pytest -m "not gpu" -q --tb=short`: `172 passed, 9 deselected, 17 warnings`.

---

# Motion Mirror v0.2a Completion

## Checklist
- [x] Fan out implementation agents for GGUF backend, hardware routing, SAM-2 propagation, and docs/tests.
- [x] Add real experimental `wan-move-gguf` backend via Diffusers GGUF loading.
- [x] Replace basic hardware auto-selection with v0.2a tier policy.
- [x] Add opt-in SAM-2 reference-video mask propagation.
- [x] Refresh README and license docs for v0.2a.
- [x] Run import checks, targeted tests, and full non-GPU pytest.

## Review
- `wan-move-gguf` is now a routed Wan backend that loads a GGUF transformer through Diffusers model-level loading and returns `GenerationResult.backend == "wan-move-gguf"`.
- `backend="auto"` now uses richer free-VRAM tiers: 8 GB VACE, 16 GB GGUF, 24 GB LightX2V fast, and 40 GB full 14B.
- `reference_masker="sam2"` adds optional reference-video mask propagation without changing the default pose-derived mask path.
- VACE mask loading now re-binarizes grayscale masks, and SAM-2 propagated masks are inverted to the existing VACE mask polarity.
- README and third-party license notes now describe the v0.2a backend surface and experimental caveats.

## Verification
- `python -m py_compile` on modified Python modules: passed.
- Pinned import checks with `PYTHONPATH=C:\Users\arnav\motion-mirror\src`: passed for package, CLI, and UI.
- Targeted v0.2a tests: `115 passed, 6 warnings`.
- Full non-GPU suite: `202 passed, 9 deselected, 17 warnings`.
- Real GPU validation was not run.
