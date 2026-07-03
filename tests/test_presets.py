"""Parity tests between repo-root presets/ and the packaged presets.

The CLI loads presets exclusively via importlib.resources from
``motion_mirror/presets`` (see cli.py). The repo-root ``presets/``
directory is only for browsing/docs and must mirror the packaged
files exactly so users never read stale values.
"""
from __future__ import annotations

import importlib.resources
from pathlib import Path

REPO_ROOT_PRESETS = Path(__file__).resolve().parent.parent / "presets"
PACKAGED_PRESETS = importlib.resources.files("motion_mirror") / "presets"


def _packaged_toml_files() -> list:
    return sorted(
        (p for p in PACKAGED_PRESETS.iterdir() if p.name.endswith(".toml") and p.is_file()),
        key=lambda p: p.name,
    )


def test_preset_filename_sets_match():
    root_names = {p.name for p in REPO_ROOT_PRESETS.glob("*.toml")}
    packaged_names = {p.name for p in _packaged_toml_files()}
    assert root_names == packaged_names, (
        f"repo-root presets/ drifted from packaged presets: "
        f"missing at root={sorted(packaged_names - root_names)}, "
        f"extra at root={sorted(root_names - packaged_names)}"
    )


def test_preset_contents_match():
    for packaged in _packaged_toml_files():
        root_file = REPO_ROOT_PRESETS / packaged.name
        assert root_file.is_file(), f"repo-root presets/ is missing {packaged.name}"
        assert root_file.read_text(encoding="utf-8") == packaged.read_text(encoding="utf-8"), (
            f"repo-root presets/{packaged.name} content differs from packaged copy"
        )
