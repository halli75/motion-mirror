"""Tests for the custom exception hierarchy and their integration points."""
from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import motion_mirror
from motion_mirror.exceptions import (
    InputError,
    MotionMirrorError,
    MultipleCharactersError,
    MultiplePeopleDetectedError,
    NoPoseDetectedError,
    SmallSubjectError,
    SmallSubjectWarning,
    SubjectError,
    UnsupportedImageError,
    UnsupportedVideoError,
    VideoDecodeError,
)


# ── Hierarchy ─────────────────────────────────────────────────────────────────

def test_all_exceptions_inherit_from_base():
    for cls in (
        InputError,
        UnsupportedImageError,
        UnsupportedVideoError,
        VideoDecodeError,
        NoPoseDetectedError,
        MultiplePeopleDetectedError,
        SubjectError,
        SmallSubjectError,
        MultipleCharactersError,
    ):
        assert issubclass(cls, MotionMirrorError), f"{cls} does not inherit MotionMirrorError"


def test_input_error_hierarchy():
    assert issubclass(UnsupportedImageError, InputError)
    assert issubclass(UnsupportedVideoError, InputError)
    assert issubclass(VideoDecodeError, InputError)


def test_pose_error_hierarchy():
    from motion_mirror.exceptions import PoseError
    assert issubclass(NoPoseDetectedError, PoseError)
    assert issubclass(MultiplePeopleDetectedError, PoseError)


def test_subject_error_hierarchy():
    assert issubclass(SmallSubjectError, SubjectError)
    assert issubclass(MultipleCharactersError, SubjectError)


def test_small_subject_warning_is_user_warning():
    assert issubclass(SmallSubjectWarning, UserWarning)


# ── Exception attributes ──────────────────────────────────────────────────────

def test_video_decode_error_stores_ffmpeg_output():
    exc = VideoDecodeError("test message", ffmpeg_output="stderr line")
    assert exc.ffmpeg_output == "stderr line"
    assert str(exc) == "test message"


def test_multiple_people_stores_count():
    exc = MultiplePeopleDetectedError("two people", count=2)
    assert exc.count == 2


def test_small_subject_stores_fraction():
    exc = SmallSubjectError("too small", bbox_fraction=0.03)
    assert abs(exc.bbox_fraction - 0.03) < 1e-9


def test_multiple_characters_stores_count():
    exc = MultipleCharactersError("two chars", count=2)
    assert exc.count == 2


# ── Public API exports ────────────────────────────────────────────────────────

def test_exceptions_exported_from_package():
    for name in (
        "MotionMirrorError",
        "InputError",
        "UnsupportedImageError",
        "UnsupportedVideoError",
        "VideoDecodeError",
        "NoPoseDetectedError",
        "MultiplePeopleDetectedError",
        "SubjectError",
        "SmallSubjectWarning",
        "SmallSubjectError",
        "MultipleCharactersError",
    ):
        assert hasattr(motion_mirror, name), f"{name} not exported from motion_mirror"


# ── Integration: segment_subject raises UnsupportedImageError ────────────────

def test_segment_unsupported_format_raises(tmp_path):
    bad_path = tmp_path / "image.bmp"
    bad_path.write_bytes(b"\x00" * 100)
    from motion_mirror.extract.segment import segment_subject
    with pytest.raises(UnsupportedImageError):
        segment_subject(bad_path)


# ── Integration: extract_pose raises UnsupportedVideoError ───────────────────

def test_pose_unsupported_format_raises(tmp_path):
    bad_path = tmp_path / "video.wmv"
    bad_path.write_bytes(b"\x00" * 100)
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.pose import extract_pose
    with pytest.raises(UnsupportedVideoError):
        extract_pose(bad_path, MotionMirrorConfig(backend="mock"))


def test_pose_missing_file_raises(tmp_path):
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.pose import extract_pose
    with pytest.raises(FileNotFoundError):
        extract_pose(tmp_path / "nonexistent.mp4", MotionMirrorConfig(backend="mock"))


def test_pose_unreadable_video_raises(tmp_path):
    """A file with wrong extension accepted by path check but rejected by cv2."""
    bad_video = tmp_path / "corrupt.mp4"
    bad_video.write_bytes(b"not a video")
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.pose import extract_pose
    with pytest.raises(VideoDecodeError):
        extract_pose(bad_video, MotionMirrorConfig(backend="mock"))


# ── Integration: FPS warning ─────────────────────────────────────────────────

def _make_video(path: Path, fps: float, frames: int = 3, size=(64, 64)) -> Path:
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 200, (h, w, 3), dtype=np.uint8))
    writer.release()
    return path


def test_low_fps_triggers_warning(tmp_path):
    vid = _make_video(tmp_path / "slow.mp4", fps=5.0)
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.pose import extract_pose
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_pose(vid, MotionMirrorConfig(backend="mock"))
    fps_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("fps" in str(w.message).lower() or "FPS" in str(w.message) for w in fps_warnings)


def test_normal_fps_no_warning(tmp_path):
    vid = _make_video(tmp_path / "normal.mp4", fps=24.0)
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.pose import extract_pose
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extract_pose(vid, MotionMirrorConfig(backend="mock"))
    fps_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "fps" in str(w.message).lower()
    ]
    assert len(fps_warnings) == 0
