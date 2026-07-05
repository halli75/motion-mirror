"""Network-gated tests for scripts/verify_model_specs.py.

These exercise the model-spec verification machinery against the LIVE Hugging
Face API, so every test is marked ``@pytest.mark.network``. Mirroring how the
repo's ``gpu`` tests self-skip at runtime (rather than via an addopts
deselection), these self-skip unless the ``network`` marker is explicitly
selected:

    pytest -m network                       # runs (hits the live HF API)
    pytest tests/test_model_specs_network.py # collected but skipped (no network)

The tests validate that the check *functions* work (repos resolve, sizes are
computed, allow_patterns coverage is complete). The authoritative pass/fail gate
on each spec's ``expected_bytes`` is the script itself
(``python scripts/verify_model_specs.py``), not these tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Every test in this module hits the network.
pytestmark = pytest.mark.network

_TIMEOUT = 30.0


def _load_verify_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_model_specs.py"
    spec = importlib.util.spec_from_file_location("verify_model_specs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_VMS = _load_verify_module()
_SPECS = _VMS._MODEL_SPECS

_ALL_KEYS = list(_SPECS)
_FILENAME_KEYS = [k for k, s in _SPECS.items() if s.get("filename") is not None]
_SNAPSHOT_KEYS = [k for k, s in _SPECS.items() if s.get("filename") is None]
_ALLOW_KEYS = [k for k, s in _SPECS.items() if s.get("allow_patterns")]


@pytest.fixture(autouse=True)
def _require_network_marker(request):
    """Skip unless the run explicitly selects the ``network`` marker.

    Mirrors the gpu tests' runtime self-skip so a plain ``pytest`` invocation
    never touches the network, while ``pytest -m network`` runs everything.
    """
    markexpr = request.config.getoption("markexpr") or ""
    if "network" not in markexpr:
        pytest.skip("network test; run explicitly with -m network")


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_repo_exists(key):
    ok, detail = _VMS.check_repo_exists(_SPECS[key]["repo_id"], _TIMEOUT)
    assert ok, f"{key}: {detail}"


@pytest.mark.parametrize("key", _FILENAME_KEYS)
def test_filename_spec_within_tolerance(key):
    ok, detail = _VMS.check_filename_spec(_SPECS[key], _TIMEOUT)
    assert ok, f"{key}: {detail}"


@pytest.mark.parametrize("key", _SNAPSHOT_KEYS)
def test_snapshot_tree_sums_positive(key):
    tree = _VMS._fetch_tree(_SPECS[key]["repo_id"], _TIMEOUT)
    files = [e for e in tree if e.get("type") == "file"]
    assert files, f"{key}: empty file tree"
    total = sum(_VMS._entry_size(e) for e in files)
    assert total > 0, f"{key}: summed size is zero"


@pytest.mark.parametrize("key", _ALLOW_KEYS)
def test_allow_patterns_coverage_complete(key):
    ok, detail = _VMS.check_snapshot_spec(_SPECS[key], _TIMEOUT)
    assert "UNCOVERED" not in detail, f"{key}: {detail}"


def test_main_runs_end_to_end():
    rc = _VMS.main(["--timeout", str(int(_TIMEOUT))])
    assert rc in (0, 1)
