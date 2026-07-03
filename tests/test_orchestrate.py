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


def test_fmt_spend_per_hr_handles_none_and_values():
    orch = _load_orchestrate_module()
    # API returns null when no pods are running -> must not crash
    assert orch._fmt_spend_per_hr(None) == "0.000"
    assert orch._fmt_spend_per_hr(0.5) == "0.500"
    assert orch._fmt_spend_per_hr(1.25) == "1.250"
