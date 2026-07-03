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

def test_passthrough_no_audio_default_returns_final_path(tmp_path):
    """Silent source, no output_path → copy lands at parent/'final.mp4'."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated)
    assert result == tmp_path / "final.mp4"


def test_passthrough_no_audio_result_exists(tmp_path):
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated)
    assert result.exists()


def test_passthrough_no_audio_custom_output_honored(tmp_path):
    """When source has no audio, the custom output_path is still written."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    custom = tmp_path / "final.mp4"
    result = passthrough_audio(source, generated, output_path=custom)
    # No audio → skip mux → copy generated to output_path
    assert result == custom
    assert custom.exists()


def test_passthrough_no_audio_copies_to_output_path(tmp_path):
    """Silent source + explicit output_path → returns output_path and the file exists."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    out = tmp_path / "outputs" / "result.mp4"
    result = passthrough_audio(source, generated, output_path=out)
    assert result == out
    assert out.exists()


def test_passthrough_no_audio_preserves_generated_evidence(tmp_path):
    """Copy, not move — generated.mp4 must survive as separate evidence."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    out = tmp_path / "outputs" / "result.mp4"
    passthrough_audio(source, generated, output_path=out)
    assert generated.exists()
    assert out.read_bytes() == generated.read_bytes()


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
    """output_path=None defaults to parent/'final.mp4' in both branches."""
    source = _make_silent_video(tmp_path / "source.mp4")
    generated = _make_silent_video(tmp_path / "generated.mp4")
    result = passthrough_audio(source, generated, output_path=None)
    # No audio in source → generated is copied to the default output path
    assert result == generated.parent / "final.mp4"
    assert result.exists()
