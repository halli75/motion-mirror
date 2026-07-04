# RunPod GPU Validation Harness

API-only (no SSH) automation for the v0.2a/v0.2b hardware validation matrix
(`docs/v02a-hardware-validation.md`). Evidence is served from the pod by
`python -m http.server 8000` and fetched through the RunPod proxy
(`https://{podId}-8000.proxy.runpod.net/...`) — RunPod has no container-log
API.

## Files

- `validate_inputs.py` — LOCAL. Picks CC-licensed smoke inputs from Wikimedia
  Commons candidates, gates them with CPU DWPose (exactly one person, big in
  frame), writes `samples.json` (URLs + sha256 + trim spec).
- `samples.json` — committed record of the chosen inputs. Pods re-fetch the
  same URLs and verify sha256.
- `pod_bootstrap.sh` — runs ON the pod (`MM_ROLE=a|b`). Role `a` (24 GB):
  vace → gguf → fast → sam2-vace smokes, Concat-ID pytest. `MM_TIER_A=1`
  runs the lean vace + gguf re-validation only. Role `b` (48 GB):
  wan-move-14b smoke. (The 12 GB-tier probe was retired 2026-07-03 after it
  measured a 15.07 GB transient peak — the tier itself was removed.)
- `orchestrate.py` — LOCAL. Launch → poll → fetch evidence → terminate.
  Spend guard tracks only pods it created (`costPerHr × uptime`), never
  account balance deltas — the account may have unrelated pods running.
- `evidence/` — pulled reports/MP4s/logs per pod.

## Run

```powershell
$env:RUNPOD_API_KEY = "..."
python runpod-validation/validate_inputs.py      # once, before pushing
python runpod-validation/orchestrate.py preflight
python runpod-validation/orchestrate.py run --role a
python runpod-validation/orchestrate.py run --role b
```

The branch must be pushed before `run` — pods `git clone -b` it.

## Safety

- `terminate` refuses pod IDs not recorded in `.orchestrator-state.json`.
- Spend guard aborts at $4.50 estimated for this session's pods.
- Wall-clock caps: role a 3.5 h, role b 2 h. One retry on spot reclaim.
