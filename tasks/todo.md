# v0.4.0a0 — retire sam2 reference-masker + add 14B VACE backends (NO GPU run)

Plan approved 2026-07-04. GPU validation deferred; local verification maximized.

## Phase 1 — delete sam2 reference-masker (parallel)
- [x] 1A src core: reference_mask.py, types.py, config.py, pipeline.py, trajectory.py
- [x] 1B cli + smoke script plumbing
- [x] 1C test pruning
- [x] 1D harness + docs
- [x] Exit gate: suite green (219) + grep sweep zero

## Phase 2 — add wan-14b-vace + wan-14b-vace-gguf (parallel, after Phase 1)
- [x] 2A config Literal + pipeline conditioning branch (!= "mock") + hardware invariant comment
- [x] 2B vace.py spec table + GGUF branch + backend-aware memory policy (+ 36 generate tests)
- [x] 2C cli download specs (3) + allow_patterns + groups (+ 9 cli tests)
- [x] 2D ui/comfy/smoke enums (+ tests)
- [x] 2E scripts/verify_model_specs.py + 16 network tests + pyproject (diffusers>=0.35.0, gguf, 0.4.0a0)
- [x] 2F harness prepared NOT run (3-smoke matrix gguf-before-full, disk 200GB, TODO(gpu-run) caps)
- [x] 2G docs (README UNVALIDATED rows, licenses)

## Phase 3 — verification (local only)
- [x] Full suite green (250 passed / 16 skipped / 7 gpu-deselected)
- [x] Grep sweep zero (reference_mask* — only evidence/ + handoff survive)
- [x] bash -n + shellcheck bootstrap
- [x] python scripts/verify_model_specs.py — 7/7 PASS live HF + pytest -m network 16 green
- [x] Wheel build 0.4.0a0 + CLI smoke
- [x] Sonnet review wave (4 reviewers); triaged; all confirmed findings fixed
- [x] handoff.md rewrite; this review section; commit + push; CI
- [x] STOP — no RunPod spend

## Review

**Scope executed**: sam2 reference-masker mode fully deleted (module, config field,
pipeline/trajectory/cli/smoke plumbing, tests, harness smoke, docs — segmenter=sam2 kept).
Two 14B backends implemented end-to-end (spec table, GGUF single-file loading, 3-way base
resolver, backend-aware memory policy, download specs with allow_patterns, groups, enums,
docs marked UNVALIDATED, harness 3-smoke matrix prepared but not run).

**Credit protection delivered**: verify_model_specs.py (7/7 live HF PASS, wired into
preflight) — immediately caught 3 wrong sizes (1.3B is 19 GB not 5; sam2 1.8 GB; 14B base
12 GB not 43). diffusers>=0.35.0 pin verified against upstream tags. Exact-args GGUF
loader tests + memory-policy matrix mutation-resistant (reviewer-confirmed non-tautological).

**Review wave (4 Sonnet reviewers)**:
- correctness: clean on NameError/resolver/offload paths; fixed stale ImportError hint,
  added up-front gguf import check, dtype now follows device.
- tests: confirmed real interception + no skip-mechanism hole; added pipeline-level
  conditioning parametrization ×3 + config construction ×5.
- docs: fixed 2 stale headings, group list, gguf license entry, diffusers Use line,
  roadmap v0.4 current, added missing LICENSE file (broken README link, pre-existing).
- harness: cache-path alignment/order/gating CONFIRMED; wired spec-verifier into
  preflight; heartbeat TODO(gpu-run) note; removed vestigial MM_GROUPS; fixed stale
  size comment.

**Final state**: 250 passed / 16 skipped / 7 deselected; wheel 0.4.0a0; shellcheck clean;
grep sweep clean; preflight includes live spec gate.

**Deferred (user decisions)**: GPU run (needs wall-cap ~6 h + spend confirm); the 11.6 GB
local GGUF CPU load-check (declined — residual GGUF-inference risk ~one partial pod run);
merge PR #7 after validation.
