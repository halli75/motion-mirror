# RunPod GPU Validation Harness

API-only (no SSH) automation for the Phase 2 VACE hardware validation
(`docs/v02a-hardware-validation.md`). Evidence is served from the pod by
`python -m http.server 8000` and fetched through the RunPod proxy
(`https://{podId}-8000.proxy.runpod.net/...`) — RunPod has no container-log
API.

## Smoke matrix (Phase 2)

A single large-disk pod runs three smokes, with downloads **interleaved** so the
order is load-bearing:

1. `wan-1.3b-vace` — pose-conditioned regression guard on the already-validated
   1.3B backend (groups `dwpose` + `vace`).
2. `wan-14b-vace-gguf` — the gguf 14B backend, run **before** the full 14B
   download. Its group (`vace-14b-gguf`) fetches only the GGUF transformer
   (~11.6 GB) and the base cache (~12 GB), so at this point the full
   `wan-14b-vace` transformer cache does **not** exist. This deliberately
   exercises the gguf resolver's base-cache fallback path — download the full
   14B first and that branch is never hit.
3. `wan-14b-vace` — the full 14B backend (group `vace-14b`, ~75 GB), fetched
   last.

Each smoke is gated on its download groups (`group_ok`); a failed group records a
failure and skips its dependents, but later smokes still attempt.

## Files

- `validate_inputs.py` — LOCAL. Picks CC-licensed smoke inputs from Wikimedia
  Commons candidates, gates them with CPU DWPose (exactly one person, big in
  frame), writes `samples.json` (URLs + sha256 + trim spec).
- `samples.json` — committed record of the chosen inputs. Pods re-fetch the
  same URLs and verify sha256.
- `pod_bootstrap.sh` — runs ON the pod (24 GB-class GPU, 200 GB container
  disk). Interleaves the `dwpose`/`vace`/`vace-14b-gguf`/`vace-14b` group
  downloads with the 3-smoke matrix above (gguf before the full 14B).
- `orchestrate.py` — LOCAL. Launch → poll → fetch evidence → terminate.
  Spend guard tracks only pods it created (`costPerHr × uptime`), never
  account balance deltas — the account may have unrelated pods running.
- `evidence/` — pulled reports/MP4s/logs per pod.

## Run

```powershell
$env:RUNPOD_API_KEY = "..."
python runpod-validation/validate_inputs.py      # once, before pushing
python runpod-validation/orchestrate.py preflight
python runpod-validation/orchestrate.py run
```

The branch must be pushed before `run` — pods `git clone -b` it.

## Safety

- `terminate` refuses pod IDs not recorded in `.orchestrator-state.json`.
- Spend guard aborts at $4.50 estimated for this session's pods.
- Wall-clock cap: 3.5 h. One retry on spot reclaim.
- Container disk: 200 GB (combined 1.3B + gguf + full 14B caches).

## Caps need a decision before the next run

The current $4.50 / 3.5 h caps were sized for the single 1.3B smoke and are
almost certainly too tight for the Phase 2 matrix: ~130 GB of downloads
(est. 45-90 min) plus two 14B smokes under sequential CPU offload
(est. 20-60 min each). A realistic budget is ~6 h wall. The values are left
unchanged and flagged with `TODO(gpu-run)` comments in `orchestrate.py` — bump
the wall cap and **re-confirm the spend cap with the user** before launching.
The harness is prepared but must NOT be run until that decision is made.
