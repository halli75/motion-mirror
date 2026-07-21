"""Local RunPod orchestrator for Motion Mirror GPU validation. No SSH.

Launches one spot pod, polls progress via the pod's http.server (through the
RunPod proxy), pulls evidence, and ALWAYS terminates the pods it created — and
only those (the account may have unrelated pods running).

Usage:
    set RUNPOD_API_KEY=...           (PowerShell: $env:RUNPOD_API_KEY="...")
    python runpod-validation/orchestrate.py preflight
    python runpod-validation/orchestrate.py run
    python runpod-validation/orchestrate.py run --experiment \
        --image char.jpg --motion motion.mp4     (private inputs, direct upload)
    python runpod-validation/orchestrate.py terminate --pod-id XXXX

If the orchestrator dies mid-run in experiment mode, upload manually once the
pod reaches the await-inputs phase:
    curl -T char.jpg   https://<pod>-8001.proxy.runpod.net/character.jpg
    curl -T motion.mp4 https://<pod>-8001.proxy.runpod.net/motion_src.mp4

Spend guard: estimates OUR spend as costPerHr x uptime per pod we created
(never account balance deltas — other projects' pods would pollute them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / ".orchestrator-state.json"
EVIDENCE_DIR = HERE / "evidence"

GRAPHQL_URL = "https://api.runpod.io/graphql"
HEADERS = {
    "User-Agent": "motion-mirror-validator/1.0",  # default urllib UA gets 403
    "Content-Type": "application/json",
    "Accept": "application/json",
}

BRANCH = "main"
REPO_URL = "https://github.com/halli75/motion-mirror.git"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

# Caps sized 2026-07-05 for the Phase 2 matrix (~118 GB downloads + two 14B
# smokes under offload). The cap is PER-RUN (run() baselines our_spend at
# start) and sits below the account balance (~$3.84) so our guard fires and
# salvages evidence before RunPod force-kills pods at $0 balance.
# 6 h on the 4090 on-demand = $2.04; the cap leaves room for one retry pod.
SPEND_CAP_USD = 3.00
# Phase 2 lineup: a single large-disk pod runs the 3-smoke VACE matrix —
# wan-1.3b-vace (regression guard), wan-14b-vace-gguf (gguf resolver base-cache
# path), and wan-14b-vace (full 14B). disk_gb sizes the container for the
# combined caches (~19 + ~24 + ~75 GB, HF-API-measured) plus image and margin.
ROLES = {
    "vace": {
        # Prefer stable Ampere/Ada datacenter cards first: RunPod's community
        # RTX 4090 batch is currently on driver 580/CUDA13.0, where our cu128
        # torch can't init CUDA (whole pool failed 2026-07-08). A6000/A40/3090
        # -class cards are managed more conservatively and still on 570/CUDA12.8.
        # 4090 kept last as a fallback. Avoid Blackwell (5090/RTX PRO/B200): new
        # arch, 580 drivers, and cu128 lacks sm_120 support.
        "gpus": [
            "NVIDIA RTX A6000",
            "NVIDIA A40",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 3090 Ti",
            "NVIDIA RTX 5000 Ada Generation",
            "NVIDIA GeForce RTX 4090",
        ],
        "disk_gb": 200,
        # 64 GB RAM: sequential offload keeps the full 14B's weights resident
        # in system RAM (bf16 transformer ~28 GB + UMT5 ~11 GB + VAE +
        # overhead ≈ 42-48 GB peak) — 48 GB was an OOM-kill risk.
        "min_ram_gb": 64,
        # 6 h wall: measured 1.3B smoke 7-26 min + est. 30-60 min downloads
        # + two untested 14B smokes under offload (est. 25-60 min each).
        "wall_cap_h": 6.0,
    },
}

# The quality experiment only downloads dwpose + the gguf-14B group (~24 GB of
# models) and runs gguf under model_cpu_offload (measured 17 GB VRAM, modest
# RAM) — far lighter than the full-14B matrix. Relax disk/RAM so the lower-stock
# stable GPUs above actually match (strict 200 GB/64 GB filters cause
# SUPPLY_CONSTRAINT on all but the 4090 pool).
EXPERIMENT_ROLE_OVERRIDES = {"disk_gb": 120, "min_ram_gb": 40}

POLL_S = 30
PROVISION_TIMEOUT_S = 15 * 60
# RunPod community 4090s are a per-host driver lottery: 570/CUDA12.8 hosts run
# our cu128 stack, 580/CUDA13.0 hosts fail torch CUDA init on the identical
# stack (observed 2026-07-08, two in a row). Such a pod aborts in ~1.5 min for
# ~$0.01, so recycle a fresh pod instead of giving up — bounded by this cap.
MAX_CUDA_RETRIES = 8
# 30 min: the heartbeat only ticks while console.log is being written, and
# the untested 75 GB 14B shard-load under offload can go legitimately silent
# for longer than the old 15 min threshold.
HEARTBEAT_STALL_S = 30 * 60


def is_stalled(last_beat_change_ts: float, now: float, threshold: float) -> bool:
    return now - last_beat_change_ts > threshold


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def gql(query: str, variables: dict | None = None, retries: int = 3) -> dict:
    """POST a GraphQL request, retrying transient network failures.

    A single dropped read must not kill a run: the run loop's finally block
    terminates the pod, so one flaky poll used to cost the whole GPU session.
    Mutations that must not double-fire (pod rent) pass retries=1.
    """
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        req = urllib.request.Request(
            GRAPHQL_URL,
            data=body,
            headers={**HEADERS, "Authorization": f"Bearer {_api_key()}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            break
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt >= max(1, retries):
                raise
            print(f"gql transient failure (attempt {attempt}/{retries}): {exc}; retrying")
            time.sleep(5 * attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError("unreachable") from last_exc
    if data.get("errors"):
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def proxy_fetch(pod_id: str, path: str, timeout: int = 30, port: int = 8000) -> bytes | None:
    url = f"https://{pod_id}-{port}.proxy.runpod.net/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _put_file(pod_id: str, local: Path, remote_name: str, retries: int = 5) -> bool:
    """HTTP PUT one file to the pod's :8001 upload receiver, with backoff."""
    url = f"https://{pod_id}-8001.proxy.runpod.net/{remote_name}"
    data = local.read_bytes()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url, data=data, method="PUT",
            headers={"User-Agent": HEADERS["User-Agent"],
                     "Content-Length": str(len(data))},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                if resp.status == 200:
                    return True
        except Exception as exc:  # noqa: BLE001
            print(f"  upload {remote_name} attempt {attempt}/{retries} failed: {exc}")
        time.sleep(5 * attempt)
    return False


def _upload_experiment_inputs(pod_id: str, experiment: dict) -> bool:
    """Upload character.jpg + motion_src.mp4, skipping any already landed."""
    ready = proxy_fetch(pod_id, "ready", timeout=15, port=8001) or b""
    ok = True
    if b"character.jpg" not in ready:
        ok = _put_file(pod_id, experiment["image"], "character.jpg") and ok
    if b"motion_src" not in ready:
        ok = _put_file(pod_id, experiment["motion"], "motion_src.mp4") and ok
    if not ok:
        return False
    ready = proxy_fetch(pod_id, "ready", timeout=15, port=8001) or b""
    return b"character.jpg" in ready and b"motion_src" in ready


def _state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"pods": {}, "spent_usd": 0.0}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ----------------------------------------------------------------- preflight


def _fmt_spend_per_hr(value: float | None) -> str:
    # API returns null when no pods are running
    return f"{0.0 if value is None else value:.3f}"


def preflight() -> None:
    me = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    print(f"balance: ${me['clientBalance']:.2f}  spend/hr (ALL pods): "
          f"${_fmt_spend_per_hr(me['currentSpendPerHr'])}")
    wanted = sorted({g for r in ROLES.values() for g in r["gpus"]})
    info = gql(
        """query { gpuTypes { id displayName memoryInGb
             lowestPrice(input:{gpuCount:1}) {
               minimumBidPrice uninterruptablePrice stockStatus } } }"""
    )["gpuTypes"]
    for g in info:
        if g["id"] in wanted:
            lp = g["lowestPrice"] or {}
            print(f"  {g['id']:32s} {g['memoryInGb']}GB  "
                  f"bid>=${lp.get('minimumBidPrice')}  "
                  f"ondemand=${lp.get('uninterruptablePrice')}  "
                  f"stock={lp.get('stockStatus')}")
    samples = json.loads((HERE / "samples.json").read_text())
    for kind in ("video", "image"):
        url = samples[kind]["url"]
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": HEADERS["User-Agent"]}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"  sample {kind}: HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            print(f"  sample {kind}: HTTP {exc.code} <-- PROBLEM")
    import subprocess

    out = subprocess.run(
        ["git", "ls-remote", "--heads", REPO_URL, BRANCH],
        capture_output=True, text=True, cwd=HERE,
    )
    print(f"  branch {BRANCH} on remote: {'YES' if out.stdout.strip() else 'NO <-- push it'}")

    # Credit protection: every model spec (repo, file, size) is verified
    # against the live HF API BEFORE a pod spends money on downloads.
    specs = subprocess.run(
        [sys.executable, str(HERE.parent / "scripts" / "verify_model_specs.py")],
        capture_output=True, text=True, cwd=HERE.parent,
        env={**os.environ, "PYTHONPATH": str(HERE.parent / "src")},
    )
    result_line = next(
        (ln.strip() for ln in specs.stdout.splitlines() if ln.strip().startswith("RESULT:")),
        "no RESULT line",
    )
    if specs.returncode == 0:
        print(f"  model specs vs HF API: {result_line}")
    else:
        print(f"  model specs vs HF API: FAILED <-- fix before launch ({result_line})")
        print(specs.stdout)


# ---------------------------------------------------------------- pod launch


def _gpu_candidates(role_cfg: dict) -> list[tuple[str, float]]:
    """All configured GPUs with any stock, preferred order, with spot bids."""
    info = gql(
        """query { gpuTypes { id
             lowestPrice(input:{gpuCount:1}) { minimumBidPrice stockStatus } } }"""
    )["gpuTypes"]
    by_id = {g["id"]: g.get("lowestPrice") or {} for g in info}
    out = []
    for gpu in role_cfg["gpus"]:
        lp = by_id.get(gpu, {})
        if lp.get("stockStatus") in ("High", "Low") and lp.get("minimumBidPrice"):
            out.append((gpu, round(float(lp["minimumBidPrice"]) * 1.4, 3)))
    if not out:
        raise RuntimeError(f"no stock for any of {role_cfg['gpus']}")
    return out


def _try_rent(mutation: str, field: str) -> dict | None:
    """One rent attempt; None on a capacity error, raise on anything else.

    retries=1: if the response is lost after the rent lands, a retry would
    silently rent a second (untracked, unguarded) pod.
    """
    try:
        return gql(mutation, retries=1)[field]
    except RuntimeError as exc:
        if "no longer any instances" in str(exc) or "SUPPLY_CONSTRAINT" in str(exc):
            return None
        raise


# Experiment-mode env: the pod trims the uploaded video to this window and runs
# a single high-quality gguf-14B generation. Kept here so launch() and tests
# share one source of truth.
EXPERIMENT_ENV = {
    "MM_EXPERIMENT": "1",
    "MM_TRIM_START": "3.0",
    "MM_TRIM_SECONDS": "5.0",
    "MM_FRAMES": "81",
    "MM_STEPS": "50",
    "MM_RESOLUTION": "480x832",
}


def _docker_args(experiment: dict | None) -> str:
    prefix = ""
    if experiment is not None:
        env = {
            **EXPERIMENT_ENV,
            "MM_IMAGE_SHA256": experiment["image_sha"],
            "MM_VIDEO_SHA256": experiment["video_sha"],
        }
        prefix = "export " + " ".join(f"{k}={v}" for k, v in env.items()) + " && "
    return (
        "bash -lc '" + prefix + "cd /workspace && "
        f"git clone --depth 1 -b {BRANCH} {REPO_URL} repo && "
        "bash repo/runpod-validation/pod_bootstrap.sh'"
    )


def launch(role: str, experiment: dict | None = None) -> str:
    cfg = {**ROLES[role], **(EXPERIMENT_ROLE_OVERRIDES if experiment else {})}
    docker_args = _docker_args(experiment)
    ports = "8000/http,8001/http" if experiment is not None else "8000/http"
    # The stock API reports GPU-level availability, but our RAM/disk filters
    # can exhaust a pool that looks "High" — so iterate every candidate GPU,
    # trying on-demand first (a spot reclaim mid-run restarts the whole
    # ~118 GB download; on-demand often costs the same as our 1.4x spot bid),
    # then spot, before giving up.
    pod = None
    for gpu, bid in _gpu_candidates(cfg):
        common_fields = f"""
            cloudType: COMMUNITY
            gpuCount: 1
            gpuTypeId: {json.dumps(gpu)}
            imageName: {json.dumps(IMAGE)}
            name: {json.dumps(f"mm-gpu-validate-{role}")}
            containerDiskInGb: {cfg["disk_gb"]}
            volumeInGb: 0
            minMemoryInGb: {cfg["min_ram_gb"]}
            dockerArgs: {json.dumps(docker_args)}
            ports: {json.dumps(ports)}
        """
        pod = _try_rent(
            f"mutation {{ podFindAndDeployOnDemand(input: {{ {common_fields} }}) "
            f"{{ id costPerHr }} }}",
            "podFindAndDeployOnDemand",
        )
        kind = "on-demand"
        if pod is None:
            print(f"{gpu}: no on-demand capacity; trying spot bid ${bid}/hr")
            pod = _try_rent(
                f"mutation {{ podRentInterruptable(input: {{ bidPerGpu: {bid} "
                f"{common_fields} }}) {{ id costPerHr }} }}",
                "podRentInterruptable",
            )
            kind = f"spot bid ${bid}/hr"
        if pod is not None:
            break
        print(f"{gpu}: no spot capacity either; trying next GPU")
    if pod is None:
        raise RuntimeError(
            f"no capacity for any of {cfg['gpus']} with "
            f"{cfg['min_ram_gb']} GB RAM / {cfg['disk_gb']} GB disk — retry later"
        )
    print(f"launched pod {pod['id']} ({gpu}, {kind}, "
          f"costPerHr ${pod.get('costPerHr')})")
    state = _state()
    state["pods"][pod["id"]] = {
        "role": role,
        "gpu": gpu,
        "cost_per_hr": float(pod.get("costPerHr") or bid),
        "launched_at": time.time(),
        "terminated": False,
    }
    _save_state(state)
    return pod["id"]


def pod_status(pod_id: str) -> dict | None:
    data = gql(
        """query($podId: String!) { pod(input:{podId:$podId}) {
             id desiredStatus costPerHr runtime { uptimeInSeconds } } }""",
        {"podId": pod_id},
    )
    return data.get("pod")


def terminate(pod_id: str) -> None:
    state = _state()
    if pod_id not in state["pods"]:
        sys.exit(f"refusing to terminate {pod_id}: not created by this orchestrator")
    try:
        gql("mutation($podId: String!) { podTerminate(input:{podId:$podId}) }",
            {"podId": pod_id})
        print(f"terminated pod {pod_id}")
    finally:
        state = _state()
        state["pods"][pod_id]["terminated"] = True
        state["pods"][pod_id].setdefault("ended_at", time.time())
        _save_state(state)


# ------------------------------------------------------------ evidence fetch


def fetch_evidence(pod_id: str, role: str) -> None:
    dest = EVIDENCE_DIR / f"pod-{role}"
    dest.mkdir(parents=True, exist_ok=True)
    paths = ["status/status.json", "status/failures.txt", "status/console.log"]
    manifest_raw = proxy_fetch(pod_id, "evidence/manifest.json")
    if manifest_raw:
        try:
            manifest = json.loads(manifest_raw)
            paths += [f["path"] for f in manifest["files"]]
        except Exception as exc:  # noqa: BLE001
            print(f"manifest unreadable ({exc}); best-effort fetch")
    else:
        print("no manifest; best-effort fetch of known paths")
        paths += [
            "evidence/env.json", "evidence/nvidia-smi.txt", "evidence/pip-freeze.txt",
            "evidence/smoke/vace.json",
            "evidence/smoke/vace-14b-gguf.json",
            "evidence/smoke/vace-14b.json",
            # Videos too: on a salvage (wall/spend/stall) the completed smokes'
            # outputs are the judgment evidence — don't leave them on the pod.
            "evidence/smoke/vace/wan-1.3b-vace/result.mp4",
            "evidence/smoke/vace-14b-gguf/wan-14b-vace-gguf/result.mp4",
            "evidence/smoke/vace-14b/wan-14b-vace/result.mp4",
            # Experiment-mode outputs (MM_EXPERIMENT=1). The pipeline writes
            # generated.mp4; the conditioning skeleton is the diagnosis artifact.
            "evidence/smoke/experiment.json",
            "evidence/smoke/experiment/wan-14b-vace-gguf/generated.mp4",
            "evidence/smoke/experiment/wan-14b-vace-gguf/conditioning_pose.mp4",
        ]
    got = 0
    for path in dict.fromkeys(paths):  # dedupe, keep order
        blob = proxy_fetch(pod_id, path, timeout=120)
        if blob is None:
            print(f"  miss: {path}")
            continue
        out = dest / path.replace("status/", "status_")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        got += 1
    print(f"fetched {got} files -> {dest}")


# -------------------------------------------------------------------- run


def our_spend(state: dict) -> float:
    # prior_spent carries real cumulative spend from pods whose exact end time
    # wasn't captured (e.g. terminated before ended_at was recorded).
    total = state.get("prior_spent", 0.0)
    now = time.time()
    for pod in state["pods"].values():
        if pod.get("ended_at"):
            end = pod["ended_at"]
        elif pod.get("terminated"):
            # Terminated but no timestamp: its real (short) cost is already in
            # prior_spent / the RunPod balance — don't count it as still running.
            continue
        else:
            end = now
        total += pod["cost_per_hr"] * max(0.0, end - pod["launched_at"]) / 3600
    return total


def run(role: str, attach_pod_id: str | None = None, experiment: dict | None = None) -> int:
    cfg = ROLES[role]
    if attach_pod_id:
        pod_id = attach_pod_id
        if pod_id not in _state()["pods"]:
            sys.exit(f"refusing to attach to {pod_id}: not created by this orchestrator")
        print(f"attached to existing pod {pod_id}")
    else:
        pod_id = launch(role, experiment=experiment)
    exit_code = 1
    retried = False
    cuda_retries = 0
    uploaded = experiment is None  # nothing to upload in the default matrix
    try:
        started = (
            _state()["pods"][pod_id]["launched_at"] if attach_pod_id else time.time()
        )
        spend_baseline = our_spend(_state())
        first_beat: float | None = None
        last_beat_change: float = time.time()
        last_beat_value: bytes | None = None
        last_phase = ""
        null_runtime_polls = 0
        while True:
            time.sleep(POLL_S)
            now = time.time()
            state = _state()
            # Guard THIS run's spend, not the state file's whole history —
            # prior sessions' pods are already paid for and counting them
            # here starves fresh runs of budget.
            spent = our_spend(state) - spend_baseline
            if spent >= SPEND_CAP_USD:
                print(f"SPEND GUARD: ${spent:.2f} >= ${SPEND_CAP_USD} — salvage + stop")
                fetch_evidence(pod_id, role)
                return 3
            if now - started > cfg["wall_cap_h"] * 3600:
                print("WALL CLOCK CAP — salvage + stop")
                fetch_evidence(pod_id, role)
                return 4

            pod = pod_status(pod_id)
            runtime_up = bool(pod and pod.get("runtime"))
            if not runtime_up:
                # The API transiently returns runtime:null on healthy pods
                # (observed 2026-07-05 on an on-demand pod mid-smoke — one
                # null poll caused a false reclaim and killed a live run).
                # The pod's own proxy is the ground truth: if it still
                # answers, the pod is alive. Only declare it dropped after
                # 3 consecutive polls where the API says null AND the proxy
                # is dead.
                proxy_alive = proxy_fetch(pod_id, "status/heartbeat", timeout=15) is not None
                if proxy_alive:
                    null_runtime_polls = 0
                    print("API reports runtime null but proxy is alive — ignoring")
                    continue
                null_runtime_polls += 1
                if first_beat is not None:
                    if null_runtime_polls < 3:
                        print(f"runtime null + proxy dead ({null_runtime_polls}/3) — confirming")
                        continue
                    print("pod runtime dropped (confirmed x3)")
                    if not retried:
                        print("retrying once with a fresh pod")
                        terminate(pod_id)
                        retried = True
                        pod_id = launch(role, experiment=experiment)
                        started, first_beat = time.time(), None
                        last_beat_value, last_beat_change = None, time.time()
                        null_runtime_polls = 0
                        uploaded = experiment is None
                        continue
                    print("already retried — giving up")
                    return 5
                if now - started > PROVISION_TIMEOUT_S:
                    print("provisioning timeout")
                    if not retried:
                        terminate(pod_id)
                        retried = True
                        pod_id = launch(role, experiment=experiment)
                        started, first_beat = time.time(), None
                        last_beat_value, last_beat_change = None, time.time()
                        uploaded = experiment is None
                        continue
                    return 5
                continue

            null_runtime_polls = 0
            beat = proxy_fetch(pod_id, "status/heartbeat", timeout=15)
            if beat is not None:
                if first_beat is None:
                    first_beat = now
                    print(f"pod is up (first heartbeat after {now - started:.0f}s)")
                if beat != last_beat_value:
                    last_beat_value, last_beat_change = beat, now
            if first_beat is not None and is_stalled(last_beat_change, now, HEARTBEAT_STALL_S):
                print(f"heartbeat stalled >{HEARTBEAT_STALL_S // 60}min — salvage + stop")
                fetch_evidence(pod_id, role)
                return 6

            status_raw = proxy_fetch(pod_id, "status/status.json", timeout=15)
            if status_raw:
                try:
                    status = json.loads(status_raw)
                except Exception:
                    continue
                if status.get("phase") != last_phase:
                    last_phase = status["phase"]
                    mins = (now - started) / 60
                    print(f"[{mins:5.1f}m ${spent:4.2f}] phase: {last_phase}  "
                          f"failures: {len(status.get('failures', []))}")
                # Experiment inputs are uploaded once the pod's receiver is up
                # (the await-inputs phase). Retried each poll until confirmed.
                if not uploaded and status.get("phase") == "await-inputs":
                    if _upload_experiment_inputs(pod_id, experiment):
                        uploaded = True
                        print("experiment inputs uploaded + confirmed on pod")
                    else:
                        print("upload incomplete; retrying next poll")
                if status.get("done"):
                    failures = status.get("failures", [])
                    # Bad-driver pod (torch can't init CUDA): cheap, pre-generation
                    # — recycle a fresh pod. NEVER retry an expensive smoke failure.
                    cuda_bad = len(failures) == 1 and any("CUDA sanity" in f for f in failures)
                    if cuda_bad and cuda_retries < MAX_CUDA_RETRIES:
                        cuda_retries += 1
                        print(f"pod GPU driver can't init CUDA — recycling fresh pod "
                              f"({cuda_retries}/{MAX_CUDA_RETRIES})")
                        terminate(pod_id)
                        pod_id = launch(role, experiment=experiment)
                        started, first_beat = time.time(), None
                        last_beat_value, last_beat_change = None, time.time()
                        last_phase, null_runtime_polls = "", 0
                        uploaded = experiment is None
                        continue
                    print("pod reports DONE — fetching evidence")
                    fetch_evidence(pod_id, role)
                    exit_code = 0 if not failures else 2
                    if failures:
                        print("failures recorded on pod:")
                        for f in failures:
                            print(f"  - {f}")
                    return exit_code
    finally:
        try:
            terminate(pod_id)
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"TERMINATE FAILED for {pod_id}: {exc} — terminate manually!")
        state = _state()
        if pod_id in state["pods"]:
            state["pods"][pod_id]["ended_at"] = time.time()
            _save_state(state)
        print(f"session spend estimate: ${our_spend(_state()):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    p_run = sub.add_parser("run")
    p_run.add_argument("--role", choices=("vace",), default="vace")
    p_run.add_argument("--experiment", action="store_true",
                       help="Quality experiment: upload --image/--motion, run gguf-14B 81f/50steps.")
    p_run.add_argument("--image", type=Path, help="Character image (experiment mode).")
    p_run.add_argument("--motion", type=Path, help="Reference motion video (experiment mode).")
    p_att = sub.add_parser("attach")
    p_att.add_argument("--role", choices=("vace",), default="vace")
    p_att.add_argument("--pod-id", required=True)
    p_term = sub.add_parser("terminate")
    p_term.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if args.cmd == "preflight":
        preflight()
        return 0
    if args.cmd == "terminate":
        terminate(args.pod_id)
        return 0
    if args.cmd == "attach":
        return run(args.role, attach_pod_id=args.pod_id)
    experiment = None
    if args.experiment:
        if not (args.image and args.motion):
            sys.exit("--experiment requires --image and --motion")
        if not args.image.exists() or not args.motion.exists():
            sys.exit(f"input not found: {args.image} / {args.motion}")
        print(f"experiment mode: hashing {args.image.name} + {args.motion.name}")
        experiment = {
            "image": args.image,
            "motion": args.motion,
            "image_sha": _sha256(args.image),
            "video_sha": _sha256(args.motion),
        }
    return run(args.role, experiment=experiment)


if __name__ == "__main__":
    raise SystemExit(main())
