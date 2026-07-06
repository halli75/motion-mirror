from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_orchestrate_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "runpod-validation" / "orchestrate.py"
    )
    spec = importlib.util.spec_from_file_location("orchestrate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_is_stalled_boundaries():
    orch = _load_orchestrate_module()
    threshold = orch.HEARTBEAT_STALL_S
    # no time elapsed -> not stalled
    assert orch.is_stalled(1000.0, 1000.0, threshold) is False
    # just under the threshold -> not stalled
    assert orch.is_stalled(1000.0, 1000.0 + threshold - 1, threshold) is False
    # exactly at the threshold -> not stalled (strict '>')
    assert orch.is_stalled(1000.0, 1000.0 + threshold, threshold) is False
    # just over the threshold -> stalled
    assert orch.is_stalled(1000.0, 1000.0 + threshold + 0.1, threshold) is True


def test_docker_args_default_has_no_experiment_exports():
    orch = _load_orchestrate_module()
    args = orch._docker_args(None)
    assert "MM_EXPERIMENT" not in args
    assert "pod_bootstrap.sh" in args


def test_docker_args_experiment_exports_env_and_shas():
    orch = _load_orchestrate_module()
    exp = {"image_sha": "aaa", "video_sha": "bbb", "image": None, "motion": None}
    args = orch._docker_args(exp)
    assert "export MM_EXPERIMENT=1" in args
    assert "MM_IMAGE_SHA256=aaa" in args
    assert "MM_VIDEO_SHA256=bbb" in args
    assert "MM_FRAMES=81" in args
    assert "MM_STEPS=50" in args
    assert "MM_RESOLUTION=480x832" in args
    # export prefix must precede the clone so the pod sees it.
    assert args.index("export MM_EXPERIMENT=1") < args.index("git clone")


def test_fmt_spend_per_hr_handles_none_and_values():
    orch = _load_orchestrate_module()
    # API returns null when no pods are running -> must not crash
    assert orch._fmt_spend_per_hr(None) == "0.000"
    assert orch._fmt_spend_per_hr(0.5) == "0.500"
    assert orch._fmt_spend_per_hr(1.25) == "1.250"


def test_gql_retries_transient_timeouts(monkeypatch):
    orch = _load_orchestrate_module()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(orch.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": {"ok": true}}'

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("read timed out")
        return _Resp()

    monkeypatch.setattr(orch.urllib.request, "urlopen", fake_urlopen)
    assert orch.gql("query { x }") == {"ok": True}
    assert calls["n"] == 3


def test_gql_single_attempt_when_retries_1(monkeypatch):
    orch = _load_orchestrate_module()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(orch.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise TimeoutError("read timed out")

    monkeypatch.setattr(orch.urllib.request, "urlopen", fake_urlopen)
    try:
        orch.gql("mutation { rent }", retries=1)
        raise AssertionError("expected TimeoutError")
    except TimeoutError:
        pass
    assert calls["n"] == 1
