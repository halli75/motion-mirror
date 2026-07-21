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
    from motion_mirror.exceptions import UnsupportedImageError
    with pytest.raises(UnsupportedImageError):
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
    cfg = MotionMirrorConfig(project_root=tmp_path / "deep", backend="mock")
    result = segment_subject(img, cfg)
    assert result.rgba_path.parent.exists()


def test_segment_excludes_soft_alpha_halo(tmp_path):
    """Anti-alias halo pixels (alpha 1..10) must NOT be promoted to solid mask.

    rembg emits a soft alpha edge around the subject; binarising with ``> 0``
    turns that halo into solid foreground. The mask should keep only genuinely
    opaque pixels (alpha > 127).
    """
    from unittest.mock import patch

    from PIL import Image

    img = _make_png(tmp_path / "char.png", size=(64, 64))

    # Controlled rembg output: solid core (alpha 255) ringed by a soft
    # anti-alias halo (alpha in 1..10), everything else transparent.
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200                       # arbitrary colour channel
    rgba[24:40, 24:40, 3] = 255               # solid core
    rgba[20:44, 20:24, 3] = 3                 # left halo band
    rgba[20:44, 40:44, 3] = 7                 # right halo band
    rgba[20:24, 20:44, 3] = 5                 # top halo band
    rgba[40:44, 20:44, 3] = 10                # bottom halo band
    fake_out = Image.fromarray(rgba, "RGBA")

    cfg = MotionMirrorConfig(project_root=tmp_path, backend="mock")
    with patch("rembg.remove", return_value=fake_out), patch(
        "motion_mirror.extract.segment._get_rembg_session", return_value=None
    ):
        result = segment_subject(img, cfg)

    # Solid core stays foreground.
    assert result.mask[32, 32] == 255
    # Soft halo bands are excluded.
    assert result.mask[22, 32] == 0, "top halo leaked into mask"
    assert result.mask[32, 22] == 0, "left halo leaked into mask"
    assert result.mask[42, 32] == 0, "bottom halo leaked into mask"
    assert set(np.unique(result.mask)).issubset({0, 255})
