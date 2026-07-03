#!/usr/bin/env bash
# Motion Mirror GPU validation — runs ON a RunPod pod, launched via dockerArgs:
#   cd /workspace && git clone -b runpod-v02a-validation <repo> repo \
#     && MM_ROLE=a bash repo/runpod-validation/pod_bootstrap.sh
#
# MM_ROLE=a : 24 GB matrix (vace, gguf, fast, sam2-vace, concat-id, tier probe)
# MM_ROLE=b : 48 GB wan-move-14b only
#
# No SSH: evidence is served by python http.server on :8000 and fetched by
# the local orchestrator through the RunPod proxy. The pod stays alive after
# DONE (sleep) so the orchestrator can pull files; the orchestrator terminates it.
set -uo pipefail

ROLE="${MM_ROLE:?set MM_ROLE=a or b}"
# MM_TIER_A=1 runs the lean re-validation set: just the three backends whose
# fixes we are confirming (vace, gguf, fast). Skips sam2-vace, concat-id (needs
# the unmerged Concat-ID fork), and the tier probe. Trims models + wall time.
TIER_A="${MM_TIER_A:-0}"
WS=/workspace
REPO=$WS/repo
RV=$REPO/runpod-validation
export HF_HOME=$WS/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WS
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
if [ "$ROLE" = a ]; then
  EXTRAS="cuda,gpu-inference,gguf,lightx2v,concat-id,dev"
else
  EXTRAS="cuda,gpu-inference,dev"
fi
if ! python3 -m pip install -e "$REPO[$EXTRAS]"; then
  record_failure "pip install extras=$EXTRAS"
  finish aborted-pip-install
fi
if [ "$ROLE" = a ] && [ "$TIER_A" != 1 ]; then
  python3 -m pip install "git+https://github.com/facebookresearch/sam2.git" \
    || record_failure "pip install sam2"
fi
# fast backend: LightX2V eagerly imports flash_attn. Install a PREBUILT wheel
# matching the image's torch 2.8 / cu12 / py3.11 — never a source build (that
# takes 30+ min). A wrong-ABI wheel installs but crashes on import, so verify
# the import and fall back to the other ABI. Best-effort: fast smoke is skipped
# if neither works.
if [ "$ROLE" = a ]; then
  # --no-deps is critical: without it pip reinstalls flash-attn's `torch`
  # dependency and clobbers the image's CUDA build, breaking cuda for ALL
  # backends. einops (its other dep) is already present via diffusers.
  fa_base="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1"
  fa_ok=false
  for abi in abiTRUE abiFALSE; do
    whl="flash_attn-2.8.3.post1+cu12torch2.8cxx11${abi}-cp311-cp311-linux_x86_64.whl"
    python3 -m pip install --no-deps "$fa_base/$whl" || continue
    if python3 -c "import flash_attn" 2>/dev/null; then fa_ok=true; break; fi
    python3 -m pip uninstall -y flash-attn >/dev/null 2>&1
  done
  $fa_ok || record_failure "flash-attn install (fast backend may fail)"
  # Guarantee the extras' CUDA torch is intact regardless of the above.
  python3 -c "import torch; assert torch.cuda.is_available()" \
    || record_failure "torch CUDA broken after flash-attn attempt"
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

# --- phase: model downloads (per group; failure skips dependent backends) ---
# NB: do not name this GROUPS — bash's special GROUPS array silently ignores
# assignments, which turns the loop variable into a group id.
if [ "$ROLE" = a ] && [ "$TIER_A" = 1 ]; then
  MM_GROUPS="dwpose light gguf fast"
elif [ "$ROLE" = a ]; then
  MM_GROUPS="dwpose light fast gguf extras identity"
else
  MM_GROUPS="dwpose wan-move"
fi
for g in $MM_GROUPS; do
  set_status "download-$g" false
  ok=false
  for attempt in 1 2; do
    if motion-mirror download --model "$g" --cache-dir $WS/mm-cache; then
      ok=true; break
    fi
    echo "download $g attempt $attempt failed; retrying"
    sleep 10
  done
  $ok || record_failure "download-$g"
done

group_ok() { ! grep -qx "download-$1" $WS/status/failures.txt 2>/dev/null; }

IMAGE=$WS/inputs/character.jpg
MOTION=$WS/inputs/motion.mp4
SMOKE=$REPO/scripts/v02a_gpu_smoke.py

run_smoke() { # run_smoke <name> <backend> [extra args...]
  local name=$1 backend=$2; shift 2
  set_status "smoke-$name" false
  python3 "$SMOKE" --image "$IMAGE" --motion "$MOTION" --backend "$backend" \
    --cache-dir $WS/mm-cache --frames 17 --density 256 \
    --output-dir $WS/evidence/smoke/$name --report $WS/evidence/smoke/$name.json \
    "$@" || record_failure "smoke-$name"
}

# --- phase: smoke matrix ---
if [ "$ROLE" = a ]; then
  group_ok dwpose && group_ok light && run_smoke vace wan-1.3b-vace \
    || record_failure "skip smoke-vace (missing models)"
  group_ok dwpose && group_ok gguf && run_smoke gguf wan-move-gguf \
    || record_failure "skip smoke-gguf (missing models)"
  group_ok dwpose && group_ok fast && run_smoke fast wan-move-fast \
    || record_failure "skip smoke-fast (missing models)"

  # Tier-A run stops here: sam2-vace, concat-id (needs the Concat-ID fork), and
  # the tier probe are out of scope for confirming the three backend fixes.
  if [ "$TIER_A" = 1 ]; then finish done-tier-a; fi

  group_ok dwpose && group_ok light && group_ok extras \
    && run_smoke sam2-vace wan-1.3b-vace --reference-masker sam2 \
    || record_failure "skip smoke-sam2-vace (missing models)"

  # --- phase: concat-id identity smoke (pytest) ---
  if group_ok dwpose && group_ok identity; then
    set_status concat-id false
    mkdir -p $WS/evidence/concat-id
    MOTION_MIRROR_GPU_CONCAT_ID=1 \
    MOTION_MIRROR_GPU_IMAGE=$IMAGE \
    MOTION_MIRROR_GPU_MOTION=$MOTION \
    MOTION_MIRROR_GPU_CACHE_DIR=$WS/mm-cache \
    MOTION_MIRROR_GPU_FRAMES=17 \
      python3 -m pytest -m gpu \
        "$REPO/tests/test_gpu_smoke_runs.py::test_concat_id_identity_gpu_pipeline_smoke" \
        -v --basetemp=$WS/evidence/concat-id --junitxml=$WS/evidence/concat-id/junit.xml \
      || record_failure "concat-id pytest"
    mkdir -p $WS/evidence/identity-comparison
    vace_mp4=$(find $WS/evidence/smoke/vace -name '*.mp4' | head -1)
    cid_mp4=$(find $WS/evidence/concat-id -name '*.mp4' | head -1)
    [ -n "$vace_mp4" ] && cp "$vace_mp4" $WS/evidence/identity-comparison/vace.mp4
    [ -n "$cid_mp4" ] && cp "$cid_mp4" $WS/evidence/identity-comparison/concat-id.mp4
    printf '%s\n' '{"note": "same image/motion/frames on both backends; qualitative identity judgment deferred to human review"}' \
      >$WS/evidence/identity-comparison/note.json
  else
    record_failure "skip concat-id (missing models)"
  fi

  # --- phase: 12GB-tier probe (t5_cpu only, no offload) ---
  if group_ok dwpose && group_ok light; then
    set_status tier-probe false
    python3 $RV/vace_tier_probe.py --image "$IMAGE" --motion "$MOTION" \
      --cache-dir $WS/mm-cache --output-dir $WS/evidence/tier-probe \
      --report $WS/evidence/tier-probe/vace-tier-probe.json \
      || record_failure "tier-probe"
  fi
else
  group_ok dwpose && group_ok wan-move && run_smoke full wan-move-14b \
    || record_failure "skip smoke-full (missing models)"
fi

finish done
