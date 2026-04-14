"""Tests for SAM-2 segmenter integration in segment_subject (Phase C).

Non-GPU tests verify the dispatch logic and rembg backward compat.
GPU tests exercise actual SAM-2 inference — require the sam2 package + CUDA.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image(path: Path, size: tuple[int, int] = (128, 128), color=(180, 120, 80)) -> Path:
    Image.new("RGB", size, color).save(str(path))
    return path


# ── rembg path: backward-compat checks ───────────────────────────────────────

def test_segment_default_uses_rembg(tmp_path):
    """Default segmenter is rembg — existing tests must still pass."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", project_root=tmp_path)
    assert cfg.segmenter == "rembg"

    img = _make_image(tmp_path / "char.png")
    result = segment_subject(img, cfg)

    assert result.rgba.shape[2] == 4
    assert result.mask.shape == result.rgba.shape[:2]
    assert result.rgba_path.exists()


def test_segment_rembg_explicit(tmp_path):
    """segmenter='rembg' explicitly should behave identically to the default."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", segmenter="rembg", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png")
    result = segment_subject(img, cfg)
    assert result.rgba_path.exists()


# ── SAM-2 dispatch: non-GPU (mocked) ─────────────────────────────────────────

def _make_sam2_mock(h: int = 64, w: int = 64):
    """Build a minimal mock that looks like SAM2ImagePredictor."""
    predictor = MagicMock()

    # masks: (3, H, W) bool;  scores: (3,) float  — three candidates
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 10:50, 10:50] = True   # candidate 0 — best
    masks[1, 20:40, 20:40] = True   # candidate 1
    masks[2, 5:60, 5:60] = True     # candidate 2 (larger)
    scores = np.array([0.95, 0.80, 0.70])

    predictor.predict.return_value = (masks, scores, None)
    predictor.set_image.return_value = None
    return predictor


def test_segment_sam2_dispatch_calls_predictor(tmp_path):
    """segmenter='sam2' should reach the SAM-2 predictor (mocked)."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", segmenter="sam2", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png", size=(64, 64))

    mock_pred = _make_sam2_mock(64, 64)

    with patch("motion_mirror.extract.segment._get_sam2_predictor", return_value=mock_pred):
        result = segment_subject(img, cfg)

    mock_pred.set_image.assert_called_once()
    mock_pred.predict.assert_called_once()

    assert result.rgba.shape == (64, 64, 4)
    assert result.mask.shape == (64, 64)
    assert result.rgba_path.exists()


def test_segment_sam2_picks_highest_score_mask(tmp_path):
    """The mask corresponding to the highest score must be selected."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", segmenter="sam2", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png", size=(64, 64))

    h, w = 64, 64
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 5:20, 5:20] = True    # small, score 0.60
    masks[1, 10:55, 10:55] = True  # medium, score 0.95  ← should be chosen
    masks[2, 1:63, 1:63] = True    # huge, score 0.40
    scores = np.array([0.60, 0.95, 0.40])

    mock_pred = MagicMock()
    mock_pred.predict.return_value = (masks, scores, None)
    mock_pred.set_image.return_value = None

    with patch("motion_mirror.extract.segment._get_sam2_predictor", return_value=mock_pred):
        result = segment_subject(img, cfg)

    # The chosen mask (index 1) should have True pixels inside [10:55, 10:55]
    assert result.mask[30, 30] == 255, "Centre pixel should be in the chosen mask"
    assert result.mask[2, 2] == 0, "Corner pixel should not be in the chosen mask"


def test_segment_sam2_warns_on_tiny_mask(tmp_path):
    """A mask covering <5 % of the image should trigger a UserWarning."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", segmenter="sam2", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png", size=(64, 64))

    h, w = 64, 64
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 30:32, 30:32] = True  # 4 pixels / 4096 ≈ 0.1 % — way under 5 %
    scores = np.array([0.95, 0.50, 0.30])

    mock_pred = MagicMock()
    mock_pred.predict.return_value = (masks, scores, None)
    mock_pred.set_image.return_value = None

    with patch("motion_mirror.extract.segment._get_sam2_predictor", return_value=mock_pred):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            segment_subject(img, cfg)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1
    assert any("5%" in str(w.message) or "centred" in str(w.message) for w in user_warnings)


def test_segment_sam2_warns_on_huge_mask(tmp_path):
    """A mask covering >85 % of the image should trigger a UserWarning."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject

    cfg = MotionMirrorConfig(backend="mock", segmenter="sam2", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png", size=(64, 64))

    h, w = 64, 64
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0] = True  # 100 % of image
    scores = np.array([0.95, 0.50, 0.30])

    mock_pred = MagicMock()
    mock_pred.predict.return_value = (masks, scores, None)
    mock_pred.set_image.return_value = None

    with patch("motion_mirror.extract.segment._get_sam2_predictor", return_value=mock_pred):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            segment_subject(img, cfg)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1
    assert any("background" in str(w.message).lower() for w in user_warnings)


def test_segment_sam2_missing_package_raises_importerror(tmp_path):
    """If sam2 is not installed, a clear ImportError must be raised."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject, _sam2_predictors

    # Clear cache to force a fresh import attempt
    _sam2_predictors.clear()

    cfg = MotionMirrorConfig(backend="mock", segmenter="sam2", project_root=tmp_path)
    img = _make_image(tmp_path / "char.png")

    with patch.dict("sys.modules", {"sam2": None, "sam2.sam2_image_predictor": None}):
        with pytest.raises(ImportError, match="sam2|SAM-2"):
            segment_subject(img, cfg)


# ── Existing segment tests still pass ────────────────────────────────────────

def test_unsupported_extension_still_raises(tmp_path):
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject
    from motion_mirror.exceptions import UnsupportedImageError

    bad = tmp_path / "image.bmp"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(UnsupportedImageError):
        segment_subject(bad, MotionMirrorConfig(project_root=tmp_path))


# ── GPU tests ─────────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_segment_sam2_real_produces_valid_mask(tmp_path):
    """Real SAM-2 inference — requires sam2 package + CUDA + HF weights."""
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.extract.segment import segment_subject, _sam2_predictors

    _sam2_predictors.clear()

    cfg = MotionMirrorConfig(
        backend="wan-move-14b", segmenter="sam2",
        device="cuda", project_root=tmp_path,
    )
    # Use a solid-colour image — SAM-2 should still return a valid mask
    img = _make_image(tmp_path / "char.png", size=(256, 256), color=(180, 120, 80))
    result = segment_subject(img, cfg)

    assert result.rgba.shape == (256, 256, 4)
    assert result.mask.shape == (256, 256)
    # Mask should contain both zeros and 255s for a real image
    assert result.mask.max() == 255
    assert result.rgba_path.exists()
    _sam2_predictors.clear()
