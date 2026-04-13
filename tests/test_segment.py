import tempfile
from pathlib import Path

import numpy as np
import pytest

from motion_mirror.config import MotionMirrorConfig
from motion_mirror.extract.segment import segment_subject
from motion_mirror.types import SegmentationResult


def _make_png(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a solid red PNG to *path*."""
    from PIL import Image

    img = Image.new("RGB", size, color=(200, 50, 50))
    img.save(str(path))
    return path


def test_segment_returns_segmentation_result(tmp_path):
    img = _make_png(tmp_path / "char.png")
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img, cfg)

    assert isinstance(result, SegmentationResult)


def test_segment_rgba_shape(tmp_path):
    img = _make_png(tmp_path / "char.png", size=(64, 64))
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img, cfg)

    assert result.rgba.shape == (64, 64, 4)
    assert result.rgba.dtype == np.uint8


def test_segment_mask_shape(tmp_path):
    img = _make_png(tmp_path / "char.png", size=(64, 64))
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img, cfg)

    assert result.mask.shape == (64, 64)
    assert result.mask.dtype == np.uint8
    # Mask values should be binary (0 or 255)
    assert set(np.unique(result.mask)).issubset({0, 255})


def test_segment_rgba_path_written(tmp_path):
    img = _make_png(tmp_path / "char.png")
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img, cfg)

    assert result.rgba_path.exists()
    assert result.rgba_path.suffix == ".png"


def test_segment_source_path_preserved(tmp_path):
    img = _make_png(tmp_path / "char.png")
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img, cfg)

    assert result.source_image_path == img


def test_segment_missing_file_raises(tmp_path):
    cfg = MotionMirrorConfig(project_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="Image not found"):
        segment_subject(tmp_path / "nonexistent.png", cfg)


def test_segment_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "file.bmp"
    bad.write_bytes(b"fake")
    cfg = MotionMirrorConfig(project_root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported image format"):
        segment_subject(bad, cfg)


def test_segment_jpeg_accepted(tmp_path):
    from PIL import Image

    img_path = tmp_path / "char.jpg"
    Image.new("RGB", (32, 32), color=(100, 200, 100)).save(str(img_path))
    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    result = segment_subject(img_path, cfg)
    assert result.rgba.shape[2] == 4


def test_segment_output_dir_created(tmp_path):
    img = _make_png(tmp_path / "char.png")
    out = tmp_path / "deep" / "outputs"
    cfg = MotionMirrorConfig(project_root=tmp_path / "deep", backend="mock")
    result = segment_subject(img, cfg)
    assert result.rgba_path.parent.exists()
