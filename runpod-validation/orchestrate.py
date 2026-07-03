"""Local RunPod orchestrator for Motion Mirror GPU validation. No SSH.

Launches one spot pod per role, polls progress via the pod's http.server
(through the RunPod proxy), pulls evidence, and ALWAYS terminates the pods
it created — and only those (the account may have unrelated pods running).

Usage:
    set RUNPOD_API_KEY=...           (PowerShell: $env:RUNPOD_API_KEY="...")
    python runpod-validation/orchestrate.py preflight
    python runpod-validation/orchestrate.py run --role a
    python runpod-validation/orchestrate.py run --role b
    python runpod-validation/orchestrate.py terminate --pod-id XXXX

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

BRANCH = "runpod-v02a-validation"
REPO_URL = "https://github.com/halli75/motion-mirror.git"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

SPEND_CAP_USD = 4.50
ROLES = {
    "a": {
        "gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"],
        "disk_gb": 140,
        "min_ram_gb": 48,
        "wall_cap_h": 3.5,
    },
    "b": {
        "gpus": ["NVIDIA A40", "NVIDIA RTX A6000"],
        "disk_gb": 70,
        "min_ram_gb": 60,
        "wall_cap_h": 2.0,
    },
}

POLL_S = 30
PROVISION_TIMEOUT_S = 15 * 60
HEARTBEAT_STALL_S = 15 * 60


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={**HEADERS, "Authorization": f"Bearer {_api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    if data.get("errors"):
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def proxy_fetch(pod_id: str, path: str, timeout: int = 30) -> bytes | None:
    url = f"https://{pod_id}-8000.proxy.runpod.net/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"pods": {}, "spent_usd": 0.0}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ----------------------------------------------------------------- preflight


def preflight() -> None:
    me = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    print(f"balance: ${me['clientBalance']:.2f}  spend/hr (ALL pods): "
          f"${me['currentSpendPerHr']:.3f}")
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


# ---------------------------------------------------------------- pod launch


def _pick_gpu(role_cfg: dict) -> tuple[str, float]:
    info = gql(
        """query { gpuTypes { id
             lowestPrice(input:{gpuCount:1}) { minimumBidPrice stockStatus } } }"""
    )["gpuTypes"]
    by_id = {g["id"]: g.get("lowestPrice") or {} for g in info}
    for gpu in role_cfg["gpus"]:
        lp = by_id.get(gpu, {})
        if lp.get("stockStatus") == "High" and lp.get("minimumBidPrice"):
            bid = round(float(lp["minimumBidPrice"]) * 1.4, 3)
            return gpu, bid
    # fall back to first GPU with any stock rather than none
    for gpu in role_cfg["gpus"]:
        lp = by_id.get(gpu, {})
        if lp.get("stockStatus") in ("High", "Low") and lp.get("minimumBidPrice"):
            bid = round(float(lp["minimumBidPrice"]) * 1.4, 3)
            print(f"WARNING: {gpu} stock is {lp['stockStatus']} — provisioning may hang")
            return gpu, bid
    raise RuntimeError(f"no stock for any of {role_cfg['gpus']}")


def launch(role: str) -> str:
    cfg = ROLES[role]
    gpu, bid = _pick_gpu(cfg)
    docker_args = (
        "bash -lc 'cd /workspace && "
        f"git clone --depth 1 -b {BRANCH} {REPO_URL} repo && "
        f"MM_ROLE={role} bash repo/runpod-validation/pod_bootstrap.sh'"
    )
    # Inline literal mutation (verified working pattern); json.dumps handles
    # GraphQL string escaping for the dockerArgs value.
    mutation = f"""
    mutation {{
      podRentInterruptable(input: {{
        bidPerGpu: {bid}
        cloudType: COMMUNITY
        gpuCount: 1
        gpuTypeId: {json.dumps(gpu)}
        imageName: {json.dumps(IMAGE)}
        name: {json.dumps(f"mm-v02a-validate-{role}")}
        containerDiskInGb: {cfg["disk_gb"]}
        volumeInGb: 0
        minMemoryInGb: {cfg["min_ram_gb"]}
        dockerArgs: {json.dumps(docker_args)}
        ports: "8000/http"
      }}) {{ id costPerHr }}
    }}"""
    pod = gql(mutation)["podRentInterruptable"]
    print(f"launched pod {pod['id']} ({gpu}, bid ${bid}/hr, "
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
            "evidence/smoke/vace.json", "evidence/smoke/gguf.json",
            "evidence/smoke/fast.json", "evidence/smoke/sam2-vace.json",
            "evidence/smoke/full.json",
            "evidence/tier-probe/vace-tier-probe.json",
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
    total = 0.0
    for pod in state["pods"].values():
        end = pod.get("ended_at") or time.time()
        total += pod["cost_per_hr"] * max(0.0, end - pod["launched_at"]) / 3600
    return total


def run(role: str, attach_pod_id: str | None = None) -> int:
    cfg = ROLES[role]
    if attach_pod_id:
        pod_id = attach_pod_id
        if pod_id not in _state()["pods"]:
            sys.exit(f"refusing to attach to {pod_id}: not created by this orchestrator")
        print(f"attached to existing pod {pod_id}")
    else:
        pod_id = launch(role)
    exit_code = 1
    retried = False
    try:
        started = (
            _state()["pods"][pod_id]["launched_at"] if attach_pod_id else time.time()
        )
        first_beat: float | None = None
        last_beat_change: float = time.time()
        last_beat_value: bytes | None = None
        last_phase = ""
        while True:
            time.sleep(POLL_S)
            now = time.time()
            state = _state()
            spent = our_spend(state)
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
                if first_beat is not None:
                    print("pod runtime dropped (spot reclaim?)")
                    if not retried:
                        print("retrying once with a fresh pod")
                        terminate(pod_id)
                        retried = True
                        pod_id = launch(role)
                        started, first_beat = time.time(), None
                        last_beat_value, last_beat_change = None, time.time()
                        continue
                    print("already retried — giving up")
                    return 5
                if now - started > PROVISION_TIMEOUT_S:
                    print("provisioning timeout")
                    if not retried:
                        terminate(pod_id)
                        retried = True
                        pod_id = launch(role)
                        started, first_beat = time.time(), None
                        last_beat_value, last_beat_change = None, time.time()
                        continue
                    return 5
                continue

            beat = proxy_fetch(pod_id, "status/heartbeat", timeout=15)
            if beat is not None:
                if first_beat is None:
                    first_beat = now
                    print(f"pod is up (first heartbeat after {now - started:.0f}s)")
                if beat != last_beat_value:
                    last_beat_value, last_beat_change = beat, now
            if first_beat is not None and now - last_beat_change > HEARTBEAT_STALL_S:
                print("heartbeat stalled >15min — salvage + stop")
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
                if status.get("done"):
                    print("pod reports DONE — fetching evidence")
                    fetch_evidence(pod_id, role)
                    exit_code = 0 if not status.get("failures") else 2
                    if status.get("failures"):
                        print("failures recorded on pod:")
                        for f in status["failures"]:
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
    p_run.add_argument("--role", choices=("a", "b"), required=True)
    p_att = sub.add_parser("attach")
    p_att.add_argument("--role", choices=("a", "b"), required=True)
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
    return run(args.role)


if __name__ == "__main__":
    raise SystemExit(main())
