"""Tests for the CLI commands (Phase 9.1).

All tests use the Typer test runner (no subprocess / no real network calls).
The run command is tested with --backend mock so no GPU or weights needed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from motion_mirror.cli import app

runner = CliRunner()


def _ascii(s: str) -> str:
    """Strip non-ASCII so Windows cp1252 terminal encoding doesn't break assertions."""
    return s.encode("ascii", errors="replace").decode("ascii")


def _make_image(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    from PIL import Image
    Image.new("RGB", size, color=(180, 120, 80)).save(str(path))
    return path


def _make_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)) -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 200, (h, w, 3), dtype=np.uint8))
    writer.release()
    return path


# ── --help ────────────────────────────────────────────────────────────────────

def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _ascii(result.output)
    for cmd in ("run", "download", "presets", "benchmark", "ui"):
        assert cmd in out, f"Command '{cmd}' missing from --help"


# ── presets ───────────────────────────────────────────────────────────────────

def test_presets_list_exit_zero():
    result = runner.invoke(app, ["presets", "--list"])
    assert result.exit_code == 0


def test_presets_list_shows_default():
    result = runner.invoke(app, ["presets", "--list"])
    assert "default" in _ascii(result.output)


def test_presets_list_shows_hq():
    result = runner.invoke(app, ["presets", "--list"])
    assert "hq" in _ascii(result.output)


def test_presets_list_shows_mock():
    result = runner.invoke(app, ["presets", "--list"])
    assert "mock" in _ascii(result.output)


# ── benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_exit_zero():
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0


def test_benchmark_shows_python_version():
    result = runner.invoke(app, ["benchmark"])
    assert "Python" in result.output


def test_benchmark_shows_platform():
    result = runner.invoke(app, ["benchmark"])
    assert "Platform" in result.output


def test_benchmark_gpu_info_no_crash():
    """--gpu-info should not crash even if torch/CUDA is absent."""
    result = runner.invoke(app, ["benchmark", "--gpu-info"])
    assert result.exit_code == 0


# ── run ───────────────────────────────────────────────────────────────────────

def test_run_mock_exits_zero(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--backend", "mock",
        "--resolution", "64x32",
        "--frames", "3",
        "--density", "16",
        "--device", "cpu",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output


def test_run_mock_prints_done(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--density", "16", "--device", "cpu",
    ])
    assert "Done" in result.output


def test_run_mock_output_file_exists(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    out_dir = tmp_path / "outputs"
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--density", "16", "--device", "cpu",
        "--output-dir", str(out_dir),
    ])
    assert result.exit_code == 0
    # Some video file must exist in output dir
    videos = list(out_dir.glob("*.mp4"))
    assert len(videos) > 0, f"No mp4 found in {out_dir}"


def test_run_with_preset_mock(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--preset", "mock",
    ])
    assert result.exit_code == 0, result.output


def test_run_missing_image_exits_nonzero(tmp_path):
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(tmp_path / "missing.png"), str(vid),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--density", "16", "--device", "cpu",
    ])
    assert result.exit_code != 0


def test_run_missing_video_exits_nonzero(tmp_path):
    img = _make_image(tmp_path / "char.png")
    result = runner.invoke(app, [
        "run", str(img), str(tmp_path / "missing.mp4"),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--density", "16", "--device", "cpu",
    ])
    assert result.exit_code != 0


def test_run_bad_preset_exits_nonzero(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--preset", "nonexistent_preset_xyz",
    ])
    assert result.exit_code != 0
