"""Tests for the CLI commands (Phase 9.1).

All tests use the Typer test runner (no subprocess / no real network calls).
The run command is tested with --backend mock so no GPU or weights needed.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from motion_mirror.cli import _MODEL_GROUPS, _MODEL_SPECS, _is_spec_cached, app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _ascii(s: str) -> str:
    """Strip non-ASCII so Windows cp1252 terminal encoding doesn't break assertions."""
    return s.encode("ascii", errors="replace").decode("ascii")


def _plain(s: str) -> str:
    """Normalize Rich/Typer help output before substring assertions."""
    return _ascii(_ANSI_RE.sub("", s))


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


def test_run_help_documents_public_surface():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    for text in (
        "wan-1.3b-vace",
        "--auto",
        "--offload-model",
        "--t5-cpu",
        "--flow-estimator",
        "--segmenter",
    ):
        assert text in out


def test_download_cache_rejects_tiny_partial_snapshot(tmp_path):
    spec = {
        "filename": None,
        "min_cached_bytes": 100,
    }
    cache_dir = tmp_path / "model"
    cache_dir.mkdir()
    (cache_dir / "partial.bin").write_bytes(b"x")

    assert not _is_spec_cached(cache_dir, spec)

    (cache_dir / "full.bin").write_bytes(b"x" * 100)
    assert _is_spec_cached(cache_dir, spec)


def test_vace_spec_counts_snapshot():
    spec = _MODEL_SPECS["wan-1.3b-vace"]
    assert spec["expected_bytes"] == 19_000_000_000
    assert spec["min_cached_bytes"] == 3_000_000_000


# ── Phase 2: 14B VACE spec shape ────────────────────────────────────────────────

def test_wan_14b_vace_spec_shape():
    spec = _MODEL_SPECS["wan-14b-vace"]
    assert spec["repo_id"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert spec["filename"] is None
    assert spec["expected_bytes"] == 76_000_000_000
    assert spec["min_cached_bytes"] == 60_000_000_000
    assert spec["cache_subdir"] == "wan-14b-vace"
    assert "allow_patterns" not in spec


def test_wan_14b_vace_gguf_spec_shape():
    spec = _MODEL_SPECS["wan-14b-vace-gguf"]
    assert spec["repo_id"] == "QuantStack/Wan2.1_14B_VACE-GGUF"
    assert spec["filename"] == "Wan2.1_14B_VACE-Q4_K_M.gguf"
    assert spec["expected_bytes"] == 11_600_000_000
    assert spec["cache_subdir"] == "wan-14b-vace-gguf"


def test_wan_14b_vace_base_spec_shape():
    spec = _MODEL_SPECS["wan-14b-vace-base"]
    assert spec["repo_id"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert spec["filename"] is None
    assert spec["expected_bytes"] == 12_000_000_000
    assert spec["min_cached_bytes"] == 9_000_000_000
    assert spec["cache_subdir"] == "wan-14b-vace-base"
    # model_index.json MUST stay in allow_patterns - the generate-side resolver's
    # completeness check keys on it.
    assert "model_index.json" in spec["allow_patterns"]


def test_download_snapshot_passes_allow_patterns(tmp_path):
    spec = _MODEL_SPECS["wan-14b-vace-base"]
    fake_usage = SimpleNamespace(free=10 ** 15)  # plenty of free space
    with patch("huggingface_hub.snapshot_download") as mock_snapshot, \
            patch("huggingface_hub.hf_hub_download"), \
            patch("motion_mirror.cli.shutil.disk_usage", return_value=fake_usage):
        result = runner.invoke(app, [
            "download", "--model", "wan-14b-vace-base",
            "--cache-dir", str(tmp_path / "cache"),
        ])
    assert result.exit_code == 0, result.output
    mock_snapshot.assert_called_once()
    _, kwargs = mock_snapshot.call_args
    assert kwargs["allow_patterns"] == spec["allow_patterns"]


def test_full_snapshot_passes_none_allow_patterns(tmp_path):
    fake_usage = SimpleNamespace(free=10 ** 15)
    with patch("huggingface_hub.snapshot_download") as mock_snapshot, \
            patch("huggingface_hub.hf_hub_download"), \
            patch("motion_mirror.cli.shutil.disk_usage", return_value=fake_usage):
        result = runner.invoke(app, [
            "download", "--model", "wan-14b-vace",
            "--cache-dir", str(tmp_path / "cache"),
        ])
    assert result.exit_code == 0, result.output
    mock_snapshot.assert_called_once()
    _, kwargs = mock_snapshot.call_args
    assert kwargs["allow_patterns"] is None


# ── Phase 2: model groups ───────────────────────────────────────────────────────

def test_vace_14b_group_exact_members():
    assert _MODEL_GROUPS["vace-14b"] == ["wan-14b-vace"]


def test_vace_14b_gguf_group_exact_members():
    assert _MODEL_GROUPS["vace-14b-gguf"] == ["wan-14b-vace-gguf", "wan-14b-vace-base"]


def test_all_group_unchanged():
    # "all" must stay the validated ~6 GB lineup - the 4 validated specs only.
    assert _MODEL_GROUPS["all"] == [
        "dwpose-pose",
        "dwpose-det",
        "wan-1.3b-vace",
        "sam2",
    ]


def test_run_help_lists_14b_backends():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "wan-14b-vace" in out
    assert "wan-14b-vace-gguf" in out


def test_download_preflight_rejects_free_space_within_margin(tmp_path):
    total_needed = _MODEL_SPECS["dwpose-pose"]["expected_bytes"]
    fake_usage = SimpleNamespace(free=int(total_needed * 1.05))
    with patch("motion_mirror.cli.shutil.disk_usage", return_value=fake_usage), \
            patch("huggingface_hub.hf_hub_download") as mock_download:
        result = runner.invoke(app, [
            "download", "--model", "dwpose-pose",
            "--cache-dir", str(tmp_path / "cache"),
        ])
    assert result.exit_code == 1, result.output
    assert "Insufficient disk space" in _plain(result.output)
    mock_download.assert_not_called()


# ── presets ───────────────────────────────────────────────────────────────────

def test_presets_list_exit_zero():
    result = runner.invoke(app, ["presets", "--list"])
    assert result.exit_code == 0


def test_presets_list_shows_default():
    result = runner.invoke(app, ["presets", "--list"])
    assert "default" in _ascii(result.output)


def test_presets_list_shows_low_vram():
    result = runner.invoke(app, ["presets", "--list"])
    assert "low-vram" in _ascii(result.output)


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


def test_run_accepts_steps_option(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--steps", "12", "--density", "16", "--device", "cpu",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output


def test_run_rejects_out_of_range_steps(tmp_path):
    img = _make_image(tmp_path / "char.png")
    vid = _make_video(tmp_path / "motion.mp4", frames=5)
    result = runner.invoke(app, [
        "run", str(img), str(vid),
        "--backend", "mock", "--resolution", "64x32",
        "--frames", "3", "--steps", "0", "--density", "16", "--device", "cpu",
    ])
    assert result.exit_code != 0


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
