#!/usr/bin/env bash
# Motion Mirror GPU validation — runs ON a RunPod pod, launched via dockerArgs:
#   cd /workspace && git clone -b main <repo> repo \
#     && bash repo/runpod-validation/pod_bootstrap.sh
#
# VACE lineup: a 3-smoke matrix on a large-disk GPU pod. Order is
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

# Fast-fail a bad-driver pod BEFORE the ~10min pip install. RunPod's community
# 4090 pool mixes 570/CUDA12.8 hosts (torch inits fine) with 580/CUDA13.0 (and
# 48GB variant) hosts where torch.cuda.is_available() is False on the identical
# stack. The base image torch already reflects the driver, so check now and
# abort in ~30s instead of ~11min. The orchestrator recycles a fresh pod on
# this failure (matches on "CUDA sanity").
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  record_failure "torch CUDA sanity check failed (bad pod driver)"
  finish aborted-cuda-sanity
fi

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

# --- phase: inputs ---
# Default path fetches the pinned public sample. MM_EXPERIMENT=1 instead
# receives the user's private media by direct upload (never committed to the
# public repo) and trims it with the MM_TRIM_* window. Both paths converge on
# inputs/character.jpg + inputs/motion.mp4.
if [ "${MM_EXPERIMENT:-0}" = "1" ]; then
  set_status await-inputs false
  if ! python3 - <<'PY'
import hashlib, http.server, os, pathlib, threading, time

inputs = pathlib.Path("/workspace/inputs")
inputs.mkdir(parents=True, exist_ok=True)
targets = {
    "/character.jpg": inputs / "character.jpg",
    "/motion_src.mp4": inputs / "motion_src",
}

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def do_PUT(self):
        dest = targets.get(self.path)
        if dest is None:
            self.send_response(404); self.end_headers(); return
        tmp = dest.parent / (dest.name + ".part")
        length = int(self.headers.get("Content-Length", 0))
        remaining = length
        with open(tmp, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk); remaining -= len(chunk)
        if length > 0 and remaining == 0:
            os.replace(tmp, dest)  # atomic: dest exists only once fully written
            self.send_response(200); self.end_headers()
        else:  # truncated / empty upload — discard so /ready never lists it
            tmp.unlink(missing_ok=True)
            self.send_response(500); self.end_headers()

    def do_GET(self):
        if self.path == "/ready":
            landed = ",".join(sorted(p.name for p in targets.values() if p.exists()))
            body = landed.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

srv = http.server.HTTPServer(("0.0.0.0", 8001), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

image, video = targets["/character.jpg"], targets["/motion_src.mp4"]
deadline = time.time() + 30 * 60
while time.time() < deadline:
    if image.exists() and video.exists():
        break
    print("waiting for inputs...", flush=True)  # keeps the heartbeat fresh
    time.sleep(30)
else:
    raise SystemExit("timed out waiting for uploaded inputs")
srv.shutdown()

def check(path, env):
    want = os.environ.get(env, "")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if want and want != got:
        raise SystemExit(f"sha256 mismatch for {path.name}: got {got} want {want}")

check(image, "MM_IMAGE_SHA256")
check(video, "MM_VIDEO_SHA256")
print("inputs received + sha256-verified")
PY
  then
    record_failure "await-inputs upload/verify"
    finish aborted-await-inputs
  fi

  set_status transcode-inputs false
  if ! python3 - <<'PY'
import os, pathlib, shutil, subprocess

inputs = pathlib.Path("/workspace/inputs")
video_src = inputs / "motion_src"
out = inputs / "motion.mp4"
start = os.environ.get("MM_TRIM_START", "3.0")
dur = os.environ.get("MM_TRIM_SECONDS", "5.0")
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
print(f"experiment input ready: motion.mp4 {n} frames")
PY
  then
    record_failure "experiment transcode"
    finish aborted-experiment-transcode
  fi
else
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
fi  # end MM_EXPERIMENT input branch

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

# The rembg segmenter fetches u2net.onnx on first use from GitHub release
# assets (release-assets.githubusercontent.com) — a host that has timed out
# mid-run (2026-07-05, killed a smoke at the segmentation step before any
# generation). Pre-warm it here with retries/backoff so the runtime call hits
# a cached model instead of a flaky network. new_session caches to ~/.u2net,
# shared with the pipeline's own new_session("u2net").
prewarm_rembg() {
  set_status prewarm-rembg false
  for attempt in 1 2 3 4; do
    if python3 -c "from rembg import new_session; new_session('u2net')"; then
      echo "rembg u2net cached"
      return 0
    fi
    echo "rembg u2net prewarm attempt $attempt failed; retrying"
    sleep 20
  done
  record_failure "prewarm-rembg"
}

IMAGE=$WS/inputs/character.jpg
MOTION=$WS/inputs/motion.mp4
SMOKE=$REPO/scripts/gpu_smoke.py

run_smoke() { # run_smoke <name> <backend> [extra args...]
  local name=$1 backend=$2; shift 2
  set_status "smoke-$name" false
  python3 "$SMOKE" --image "$IMAGE" --motion "$MOTION" --backend "$backend" \
    --cache-dir $WS/mm-cache \
    --output-dir "$WS/evidence/smoke/$name" --report "$WS/evidence/smoke/$name.json" \
    "$@" || record_failure "smoke-$name"
}

if [ "${MM_EXPERIMENT:-0}" = "1" ]; then
  # Quality experiment: only dwpose + the gguf-14B backend, then a single
  # full-length high-step run on the uploaded inputs.
  echo "experiment matrix: dwpose + vace-14b-gguf -> single wan-14b-vace-gguf run"
  download_group dwpose
  download_group vace-14b-gguf
  prewarm_rembg
  rembg_ok() { ! grep -qx "prewarm-rembg" $WS/status/failures.txt 2>/dev/null; }
  if group_ok dwpose && group_ok vace-14b-gguf && rembg_ok; then
    run_smoke experiment wan-14b-vace-gguf \
      --frames "${MM_FRAMES:-81}" --steps "${MM_STEPS:-50}" \
      --resolution "${MM_RESOLUTION:-480x832}" --density "${MM_DENSITY:-256}"
  else
    record_failure "skip smoke-experiment (missing models)"
  fi
else
  # (1) regression guard: 1.3B VACE on the already-validated backend.
  download_group dwpose
  download_group vace
  if group_ok dwpose && group_ok vace; then
    run_smoke vace wan-1.3b-vace --frames 17 --density 256
    # (1b) no-offload case: exercises the attention-slicing VRAM gate
    # (vace.py:_needs_attention_slicing), which only skips slicing when
    # offload_model=False and free VRAM clears the threshold. The rest of
    # the matrix always runs with offload_model=True and never hits this
    # branch, so this is the only smoke case that covers it.
    run_smoke vace-no-offload wan-1.3b-vace --frames 17 --density 256 --no-offload-model
  else
    record_failure "skip smoke-vace (missing models)"
  fi

  # (2) gguf 14B BEFORE the full 14B download — the full transformer cache does
  #     not exist yet, so this covers the gguf resolver's base-cache fallback.
  download_group vace-14b-gguf
  if group_ok dwpose && group_ok vace-14b-gguf; then
    run_smoke vace-14b-gguf wan-14b-vace-gguf --frames 17 --density 256
  else
    record_failure "skip smoke-vace-14b-gguf (missing models)"
  fi

  # (3) full 14B VACE.
  download_group vace-14b
  if group_ok dwpose && group_ok vace-14b; then
    run_smoke vace-14b wan-14b-vace --frames 17 --density 256
  else
    record_failure "skip smoke-vace-14b (missing models)"
  fi
fi

finish "done"
