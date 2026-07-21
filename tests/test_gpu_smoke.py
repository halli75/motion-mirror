from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_smoke_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("gpu_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_config_uses_supported_motion_mirror_config_fields(tmp_path):
    smoke = _load_smoke_module()

    cfg = smoke._build_config(
        backend="wan-1.3b-vace",
        output_dir=tmp_path / "smoke" / "vace",
        cache_dir=tmp_path / "cache",
        resolution="832x480",
        frames=17,
        steps=50,
        density=256,
        device="cpu",
    )

    assert cfg.project_root == tmp_path / "smoke"
    assert cfg.output_dir_name == "vace"
    assert cfg.output_dir == tmp_path / "smoke" / "vace"
    assert cfg.cache_dir == tmp_path / "cache"
    assert cfg.backend == "wan-1.3b-vace"
    assert cfg.num_frames == 17
    assert cfg.num_inference_steps == 50
    assert cfg.trajectory_density == 256


# --- content-gate fixtures -------------------------------------------------

_SIZE = 160


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 8.0) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), f"could not open VideoWriter for {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _draw_skeleton(canvas: np.ndarray, offset: int) -> None:
    """Thin bright polylines mimicking an OpenPose stick figure."""
    white = (255, 255, 255)
    cv2.circle(canvas, (80 + offset, 30), 10, white, 2)
    cv2.line(canvas, (80 + offset, 40), (80 + offset, 100), white, 2)
    cv2.line(canvas, (80 + offset, 55), (50 + offset, 80), white, 2)
    cv2.line(canvas, (80 + offset, 55), (110 + offset, 80), white, 2)
    cv2.line(canvas, (80 + offset, 100), (55 + offset, 140), white, 2)
    cv2.line(canvas, (80 + offset, 100), (105 + offset, 140), white, 2)


def _skeleton_frames(background_bgr: tuple[int, int, int], count: int = 8) -> list[np.ndarray]:
    frames = []
    for i in range(count):
        canvas = np.full((_SIZE, _SIZE, 3), background_bgr, dtype=np.uint8)
        _draw_skeleton(canvas, offset=3 * i)
        frames.append(canvas)
    return frames


def _subject_frames(count: int = 8, moving: bool = True) -> list[np.ndarray]:
    """Dark background + a filled moving blob 'person' covering ~15% of the frame."""
    frames = []
    for i in range(count):
        canvas = np.full((_SIZE, _SIZE, 3), 5, dtype=np.uint8)
        offset = 3 * i if moving else 0
        cv2.ellipse(canvas, (60 + offset, 80), (28, 45), 0, 0, 360, (70, 70, 70), -1)
        frames.append(canvas)
    return frames


# --- content-gate behaviour ------------------------------------------------


def test_gate_rejects_skeleton_on_colored_background(tmp_path):
    smoke = _load_smoke_module()
    output = _write_video(
        tmp_path / "output.mp4", _skeleton_frames(background_bgr=(180, 120, 60))
    )
    control = _write_video(
        tmp_path / "conditioning_pose.mp4", _skeleton_frames(background_bgr=(0, 0, 0))
    )

    ok, reason = smoke._check_content(output, control)

    assert not ok, f"skeleton-on-color passed the gate: {reason}"
    assert "fill ratio" in reason, reason


def test_gate_passes_dark_subject_without_control(tmp_path):
    smoke = _load_smoke_module()
    output = _write_video(tmp_path / "output.mp4", _subject_frames())

    ok, reason = smoke._check_content(output, None)

    assert ok, f"real dark-background subject rejected: {reason}"


def test_gate_rejects_static_output(tmp_path):
    smoke = _load_smoke_module()
    output = _write_video(tmp_path / "output.mp4", _subject_frames(moving=False))

    ok, reason = smoke._check_content(output, None)

    assert not ok
    assert "static" in reason, reason


def test_control_passthrough_detected_across_resolution_mismatch():
    smoke = _load_smoke_module()
    frames = [f.astype("float32") for f in _skeleton_frames(background_bgr=(180, 120, 60))]
    # Control at half resolution: the compare must resize, not broadcast-crash.
    control = [
        cv2.resize(f, (_SIZE // 2, _SIZE // 2)).astype("float32")
        for f in _skeleton_frames(background_bgr=(0, 0, 0))
    ]

    ok, reason = smoke._check_control_passthrough(frames, control)

    assert not ok
    assert "edge IoU" in reason, reason


def test_control_passthrough_allows_distinct_subject():
    smoke = _load_smoke_module()
    frames = [f.astype("float32") for f in _subject_frames()]
    control = [f.astype("float32") for f in _skeleton_frames(background_bgr=(0, 0, 0))]

    ok, _ = smoke._check_control_passthrough(frames, control)

    assert ok
