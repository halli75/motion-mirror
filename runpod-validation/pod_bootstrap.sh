#!/usr/bin/env bash
# Motion Mirror GPU validation — runs ON a RunPod pod, launched via dockerArgs:
#   cd /workspace && git clone -b runpod-v02a-validation <repo> repo \
#     && bash repo/runpod-validation/pod_bootstrap.sh
#
# VACE lineup (Phase 2): a 3-smoke matrix on a large-disk GPU pod. Order is
# load-bearing — downloads are interleaved with smokes:
#   (1) dwpose + vace  -> smoke wan-1.3b-vace       (pose-conditioned regression
#       guard on the already-validated backend);
#   (2) vace-14b-gguf  -> smoke wan-14b-vace-gguf   (run BEFORE the full 14B
#       download so the gguf resolver falls back to the base cache — the full
#       transformer cache does NOT exist yet, so this covers that resolver path);
#   (3) vace-14b       -> smoke wan-14b-vace        (full 14B).
#
# No SSH: evidence is served by python http.server on :8000 and fetched by
# the local orchestrator through the RunPod proxy. The pod stays alive after
# DONE (sleep) so the orchestrator can pull files; the orchestrator terminates it.
set -uo pipefail

ROLE=vace
WS=/workspace
REPO=$WS/repo
export HF_HOME=$WS/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WS" || exit 1
mkdir -p status evidence evidence/smoke inputs mm-cache "$HF_HOME"

# --- observability first: http server + heartbeat + tee'd console ---
nohup python3 -m http.server 8000 --directory $WS >$WS/status/http.log 2>&1 &
# Heartbeat ticks only while console.log is actively growing (progress bars /
# step logs keep its mtime fresh during real work). A hard hang freezes the
# log, the heartbeat goes stale, and the orchestrator's stall guard can fire.
touch $WS/status/console.log
( while true; do
    now=$(date -u +%s)
    log_mtime=$(stat -c %Y $WS/status/console.log 2>/dev/null || echo 0)
    if [ $((now - log_mtime)) -le 120 ]; then echo "$now" >$WS/status/heartbeat; fi
    sleep 20
  done ) &
exec > >(tee -a $WS/status/console.log) 2>&1

record_failure() {
  echo "$1" >>$WS/status/failures.txt
  echo "FAILURE RECORDED: $1"
}

set_status() { # set_status <phase> <done:true|false>
  python3 - "$ROLE" "$1" "$2" <<'PY'
import json, sys, time, pathlib
role, phase, done = sys.argv[1:4]
failures = pathlib.Path("/workspace/status/failures.txt")
fails = failures.read_text().splitlines() if failures.exists() else []
pathlib.Path("/workspace/status/status.json").write_text(json.dumps({
    "role": role, "phase": phase, "done": done == "true",
    "updated_at": int(time.time()), "failures": fails,
}, indent=2))
PY
  echo "=== PHASE: $1 ==="
}

finish() { # finish <last-phase-label>
  python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path("/workspace/evidence")
files = [
    {"path": str(p.relative_to("/workspace")), "bytes": p.stat().st_size}
    for p in sorted(root.rglob("*")) if p.is_file()
]
(root / "manifest.json").write_text(json.dumps({"files": files}, indent=2))
PY
  set_status "$1" true
  echo "DONE — sleeping so the orchestrator can fetch evidence"
  sleep 7200
  exit 0
}

# --- phase: env info ---
set_status env-info false
nvidia-smi >$WS/evidence/nvidia-smi.txt 2>&1 || record_failure "nvidia-smi failed"
python3 - <<'PY'
import json, platform, sys
info = {"platform": platform.platform(), "python": sys.version}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "total_memory_gb": round(p.total_memory / 1024**3, 2)}
except Exception as exc:
    info["torch_error"] = str(exc)
open("/workspace/evidence/env.json", "w").write(json.dumps(info, indent=2))
PY

# --- phase: pip install (ABORT on failure) ---
set_status pip-install false
python3 -m pip install -U pip >/dev/null 2>&1
EXTRAS="cuda,gpu-inference,dev"
if ! python3 -m pip install -e "${REPO}[$EXTRAS]"; then
  record_failure "pip install extras=$EXTRAS"
  finish aborted-pip-install
fi
# Pin the ML stack to the last-known-good set (pip-freeze of the 2026-07-04
# passing pod, plus the gguf version verified current at prep time). An
# unpinned "latest diffusers" on rent day is exactly the kind of churn that
# breaks WanVACE/GGUF loading mid-run and wastes the pod.
if ! python3 -m pip install "diffusers==0.39.0" "transformers==5.13.0" \
    "accelerate==1.14.0" "gguf==0.19.0"; then
  record_failure "pip pin known-good ML stack"
  finish aborted-pip-install
fi
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA gone after installs'"; then
  record_failure "torch CUDA sanity check failed after pip installs"
  finish aborted-cuda-sanity
fi
python3 -m pip freeze >$WS/evidence/pip-freeze.txt

# --- phase: samples (ABORT on integrity failure) ---
set_status samples false
if ! python3 - <<'PY'
import hashlib, json, pathlib, shutil, subprocess, sys, urllib.request

samples = json.loads(pathlib.Path("/workspace/repo/runpod-validation/samples.json").read_text())
inputs = pathlib.Path("/workspace/inputs")

def fetch(spec, dest):
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "motion-mirror-validator/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha != spec["sha256"]:
        raise SystemExit(f"sha256 mismatch for {spec['url']}: {sha}")

video_src = inputs / "motion_src"
image = inputs / "character.jpg"
fetch(samples["video"], video_src)
fetch(samples["image"], image)

# Pipeline accepts .mp4/.mov/.avi/.mkv only -> trim + transcode.
start = samples["video"].get("trim_start_seconds", 0)
dur = samples["trim_seconds"]
out = inputs / "motion.mp4"
base = ["-y", "-ss", str(start), "-t", str(dur), "-i", str(video_src), "-an"]
cmds = []
ff = shutil.which("ffmpeg")
if ff:
    cmds.append([ff, *base, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    cmds.append([ff, *base, "-c:v", "mpeg4", "-q:v", "3", str(out)])
sff = shutil.which("static_ffmpeg")
if sff:
    cmds.append([sff, *base, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    cmds.append([sff, *base, "-c:v", "mpeg4", "-q:v", "3", str(out)])
for cmd in cmds:
    if subprocess.run(cmd).returncode == 0 and out.exists() and out.stat().st_size > 0:
        break
else:
    raise SystemExit("all transcode attempts failed")

import cv2
cap = cv2.VideoCapture(str(out))
ok, _ = cap.read()
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if not ok or n < 17:
    raise SystemExit(f"transcoded motion.mp4 unreadable or too short ({n} frames)")
print(f"samples ready: motion.mp4 {n} frames")
PY
then
  record_failure "sample fetch/verify/transcode"
  finish aborted-samples
fi

# --- model downloads + smoke matrix (interleaved; failure skips dependents) ---
# NB: do not name this GROUPS — bash's special GROUPS array silently ignores
# assignments, which turns the loop variable into a group id.
# Ordered download set (canonical list for the run):
#   dwpose         = pose extractor;   vace = wan-1.3b-vace weights;
#   vace-14b-gguf  = wan-14b-vace-gguf (~11.6 GB) + wan-14b-vace-base (~12 GB);
#   vace-14b       = wan-14b-vace full transformer (~75 GB).
# Downloads are NOT done all-up-front: the full 14B (vace-14b) is fetched only
# AFTER the gguf smoke, so that smoke exercises the gguf resolver's base-cache
# fallback while the full transformer cache is still absent. The order is the
# explicit download_group/run_smoke call sequence below — there is no list
# variable to edit.
echo "download+smoke matrix: dwpose+vace -> smoke vace -> vace-14b-gguf -> smoke gguf -> vace-14b -> smoke 14b"

download_group() { # download_group <group>
  local g=$1 ok=false
  set_status "download-$g" false
  for attempt in 1 2; do
    if motion-mirror download --model "$g" --cache-dir $WS/mm-cache; then
      ok=true; break
    fi
    echo "download $g attempt $attempt failed; retrying"
    sleep 10
  done
  $ok || record_failure "download-$g"
}

group_ok() { ! grep -qx "download-$1" $WS/status/failures.txt 2>/dev/null; }

IMAGE=$WS/inputs/character.jpg
MOTION=$WS/inputs/motion.mp4
SMOKE=$REPO/scripts/v02a_gpu_smoke.py

run_smoke() { # run_smoke <name> <backend> [extra args...]
  local name=$1 backend=$2; shift 2
  set_status "smoke-$name" false
  python3 "$SMOKE" --image "$IMAGE" --motion "$MOTION" --backend "$backend" \
    --cache-dir $WS/mm-cache --frames 17 --density 256 \
    --output-dir "$WS/evidence/smoke/$name" --report "$WS/evidence/smoke/$name.json" \
    "$@" || record_failure "smoke-$name"
}

# (1) regression guard: 1.3B VACE on the already-validated backend.
download_group dwpose
download_group vace
if group_ok dwpose && group_ok vace; then
  run_smoke vace wan-1.3b-vace
else
  record_failure "skip smoke-vace (missing models)"
fi

# (2) gguf 14B BEFORE the full 14B download — the full transformer cache does
#     not exist yet, so this covers the gguf resolver's base-cache fallback.
download_group vace-14b-gguf
if group_ok dwpose && group_ok vace-14b-gguf; then
  run_smoke vace-14b-gguf wan-14b-vace-gguf
else
  record_failure "skip smoke-vace-14b-gguf (missing models)"
fi

# (3) full 14B VACE.
download_group vace-14b
if group_ok dwpose && group_ok vace-14b; then
  run_smoke vace-14b wan-14b-vace
else
  record_failure "skip smoke-vace-14b (missing models)"
fi

finish "done"
