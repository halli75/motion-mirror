# RunPod GPU Validation Harness

API-only (no SSH) automation for validating the VACE backends on real GPUs.
The pod serves progress + evidence over `python -m http.server` through the
RunPod proxy (`https://{podId}-8000.proxy.runpod.net/...`); the local
orchestrator polls it, pulls results, and always terminates the pods it created.

## Files

- `orchestrate.py` — LOCAL. `preflight` → `run` → fetch evidence → `terminate`.
  Prefers stable Ampere/Ada datacenter GPUs, auto-recycles pods that can't init
  CUDA (a driver lottery on some community cards), and guards spend per-pod
  (`costPerHr × uptime`, never account-balance deltas).
- `pod_bootstrap.sh` — runs ON the pod. Installs the pinned stack, downloads
  weights, and runs the smoke matrix. `MM_EXPERIMENT=1` switches to a single
  high-quality run on privately uploaded inputs.
- `validate_inputs.py` — LOCAL. Picks CC-licensed smoke inputs, gates them with
  CPU DWPose (one person, large in frame), writes `samples.json`.
- `samples.json` — the chosen public smoke inputs (URLs + sha256 + trim).

## Run

```powershell
$env:RUNPOD_API_KEY = "..."
python runpod-validation/orchestrate.py preflight       # balance, stock, spec check
python runpod-validation/orchestrate.py run             # default smoke matrix

# Quality run on your own inputs (uploaded to the pod, never committed):
python runpod-validation/orchestrate.py run --experiment `
  --image character.jpg --motion motion.mp4
```

The branch must be pushed first — pods `git clone -b` it.

## Safety

- `terminate` refuses any pod ID not in `.orchestrator-state.json`.
- Per-run spend guard + wall-clock cap salvage evidence before stopping.
- Bad-driver pods fail fast (~30 s) and are recycled automatically.
- Private `--experiment` inputs upload to the pod over an sha256-verified
  channel and are never written to this (public) repo — pod evidence is
  gitignored.
