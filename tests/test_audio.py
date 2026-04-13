"""Tests for audio passthrough post-processing (Phase 7).

All tests are CPU-only and do not require a GPU.
Silent synthetic MP4s are used throughout.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_mirror.postprocess.audio import passthrough_audio


def _make_silent_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a silent synthetic MP4 using cv2.VideoWriter."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (w, h))
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return path


# ── no-audio path ─────────────────────────────────────────────────────────────

def test_passthrough_no_audio_returns_generated_path(tmp_path):
    """Silent source → generated video returned unchanged."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated)
    assert result == generated


def test_passthrough_no_audio_result_exists(tmp_path):
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated)
    assert result.exists()


def test_passthrough_no_audio_custom_output_not_used(tmp_path):
    """When source has no audio, custom output_path is NOT written — generated is returned."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    custom = tmp_path / "final.mp4"
    result = passthrough_audio(source, generated, output_path=custom)
    # No audio → skip mux → return generated as-is
    assert result == generated
    assert not custom.exists()


# ── validation ────────────────────────────────────────────────────────────────

def test_passthrough_missing_source_raises(tmp_path):
    generated = _make_silent_video(tmp_path / "generated.mp4")
    with pytest.raises(FileNotFoundError, match="Source video not found"):
        passthrough_audio(tmp_path / "nonexistent.mp4", generated)


def test_passthrough_missing_generated_raises(tmp_path):
    source = _make_silent_video(tmp_path / "source.mp4")
    with pytest.raises(FileNotFoundError, match="Generated video not found"):
        passthrough_audio(source, tmp_path / "nonexistent.mp4")


# ── return type ───────────────────────────────────────────────────────────────

def test_passthrough_returns_path_not_tuple(tmp_path):
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated)
    assert isinstance(result, Path)


# ── default output path ───────────────────────────────────────────────────────

def test_passthrough_default_output_path_name(tmp_path):
    """If audio were present, output defaults to parent/'final.mp4'.
    We verify the logic by checking the no-audio short-circuit still returns
    the generated path (not a 'final.mp4' that was never created)."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated, output_path=None)
    # No audio in source → passthrough returns generated_video_path
    assert result == generated
