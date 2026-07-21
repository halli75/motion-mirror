"""Validation of the packaged generation presets.

The CLI loads presets exclusively via importlib.resources from
``motion_mirror/presets`` (see cli.py).
"""
from __future__ import annotations

import importlib.resources
import tomllib

PACKAGED_PRESETS = importlib.resources.files("motion_mirror") / "presets"

REQUIRED_FIELDS = ("name", "description", "backend", "resolution", "num_frames")


def _packaged_toml_files() -> list:
    return sorted(
        (p for p in PACKAGED_PRESETS.iterdir() if p.name.endswith(".toml") and p.is_file()),
        key=lambda p: p.name,
    )


def test_presets_exist():
    names = {p.name for p in _packaged_toml_files()}
    assert {"default.toml", "low-vram.toml", "mock.toml"} <= names


def test_presets_parse_with_required_fields():
    for packaged in _packaged_toml_files():
        data = tomllib.loads(packaged.read_text(encoding="utf-8"))
        assert "preset" in data, f"{packaged.name} is missing the [preset] table"
        preset = data["preset"]
        for field in REQUIRED_FIELDS:
            assert field in preset, f"{packaged.name} is missing '{field}'"
        assert preset["name"] == packaged.name.removesuffix(".toml")


def test_vace_presets_satisfy_frame_constraint():
    """The real VACE path requires num_frames = 4k+1; shipped presets must comply."""
    for packaged in _packaged_toml_files():
        preset = tomllib.loads(packaged.read_text(encoding="utf-8"))["preset"]
        if preset["backend"] == "mock":
            continue
        frames = preset["num_frames"]
        assert (frames - 1) % 4 == 0, (
            f"{packaged.name}: num_frames={frames} violates the VACE 4k+1 frame rule"
        )
