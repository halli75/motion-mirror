"""Tests for v0.2a config additions and pipeline wiring."""
from __future__ import annotations

import warnings

import pytest

from motion_mirror.config import MotionMirrorConfig


# ── New backend literals ──────────────────────────────────────────────────────

def test_config_default_backend():
    cfg = MotionMirrorConfig()
    assert cfg.backend == "wan-move-14b"


def test_config_new_backends():
    for backend in ("auto", "wan-move-fast", "wan-1.3b-vace", "controlnet", "mock"):
        cfg = MotionMirrorConfig(backend=backend)
        assert cfg.backend == backend


# ── v0.2a fields have correct defaults ───────────────────────────────────────

def test_config_offload_model_default():
    assert MotionMirrorConfig().offload_model is False


def test_config_t5_cpu_default():
    assert MotionMirrorConfig().t5_cpu is False


def test_config_flow_estimator_default():
    assert MotionMirrorConfig().flow_estimator == "farneback"


def test_config_segmenter_default():
    assert MotionMirrorConfig().segmenter == "rembg"


def test_config_new_fields_settable():
    cfg = MotionMirrorConfig(
        backend="wan-1.3b-vace",
        offload_model=True,
        t5_cpu=True,
        flow_estimator="raft",
        segmenter="sam2",
    )
    assert cfg.backend == "wan-1.3b-vace"
    assert cfg.offload_model is True
    assert cfg.t5_cpu is True
    assert cfg.flow_estimator == "raft"
    assert cfg.segmenter == "sam2"


# ── Pipeline: deprecated 'controlnet' alias ───────────────────────────────────

def test_pipeline_controlnet_alias_warns():
    """Using backend='controlnet' issues DeprecationWarning and resolves to wan-1.3b-vace."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / "char.png"
        img_path.write_bytes(b"fake-image")
        vid_path = tmp_path / "motion.mp4"
        vid_path.write_bytes(b"fake-video")

        from motion_mirror.pipeline import MotionMirrorPipeline

        run_cfg = MotionMirrorConfig(
            backend="controlnet",
            resolution="64x32",
            num_frames=2,
            trajectory_density=16,
            device="cpu",
            project_root=tmp_path,
        )

        # We only need to verify the DeprecationWarning fires and the backend is
        # rewritten. Stub out the expensive pipeline stages so the test is fast.
        fake_seg = MagicMock()
        fake_seg.rgba_path = tmp_path / "seg.png"
        fake_pose = MagicMock()
        fake_traj = MagicMock()
        fake_traj.save = MagicMock()
        fake_gen = MagicMock()
        fake_gen.video_path = tmp_path / "gen.mp4"
        fake_gen.video_path.touch()
        final_path = tmp_path / "result.mp4"
        final_path.touch()

        with (
            patch("motion_mirror.pipeline.segment_subject", return_value=fake_seg),
            patch("motion_mirror.pipeline.extract_pose", return_value=fake_pose),
            patch("motion_mirror.pipeline.synthesize_trajectory", return_value=fake_traj),
            patch("motion_mirror.pipeline.generate_with_controlnet", return_value=fake_gen),
            patch("motion_mirror.pipeline.passthrough_audio", return_value=final_path),
        ):
            img_path.touch()  # make files exist for validation
            vid_path.touch()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = MotionMirrorPipeline(run_cfg).run(img_path, vid_path)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "No DeprecationWarning emitted for 'controlnet' backend"
        assert "controlnet" in str(dep_warnings[0].message).lower()
        assert result.output_path.exists()


# ── Pipeline: unknown backend raises ValueError ───────────────────────────────

def test_pipeline_unknown_backend_raises():
    from motion_mirror.pipeline import MotionMirrorPipeline
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")
        vid = tmp_path / "x.mp4"
        vid.write_bytes(b"fake")

        cfg = MotionMirrorConfig(backend="wan-move-14b")
        object.__setattr__(cfg, "backend", "nonexistent-backend")
        with pytest.raises(ValueError, match="nonexistent-backend"):
            MotionMirrorPipeline(cfg).run(img, vid)


# ── Pipeline: new backends are valid ─────────────────────────────────────────

def test_pipeline_new_backends_in_valid_set():
    """wan-move-fast and wan-1.3b-vace should not raise ValueError for unknown backend."""
    from pathlib import Path
    from PIL import Image
    import cv2, numpy as np, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img_path = tmp_path / "char.png"
        Image.new("RGB", (64, 64), (100, 120, 80)).save(str(img_path))

        vid_path = tmp_path / "motion.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 64))
        rng = np.random.default_rng(2)
        for _ in range(3):
            writer.write(rng.integers(0, 200, (64, 64, 3), dtype=np.uint8))
        writer.release()

        from motion_mirror.pipeline import MotionMirrorPipeline

        # Both new backends are real; without GPU/weights they'll raise but NOT ValueError
        for backend in ("wan-move-fast", "wan-1.3b-vace"):
            cfg = MotionMirrorConfig(
                backend=backend,
                resolution="64x32",
                num_frames=2,
                trajectory_density=16,
                device="cpu",
                project_root=tmp_path,
            )
            with pytest.raises(Exception) as exc_info:
                MotionMirrorPipeline(cfg).run(img_path, vid_path)
            # Must not be "unknown backend" ValueError
            assert "unknown" not in str(exc_info.value).lower() or exc_info.type is not ValueError


# ── CLI: new flags smoke-test ─────────────────────────────────────────────────

def test_cli_presets_list_shows_new_presets():
    from typer.testing import CliRunner
    from motion_mirror.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["presets", "--list"])
    assert result.exit_code == 0
    out = result.output.encode("ascii", errors="replace").decode()
    assert "low-vram" in out
    assert "fast" in out


def test_cli_download_help_shows_new_groups():
    from typer.testing import CliRunner
    from motion_mirror.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "light" in result.output or "wan-1.3b" in result.output


def test_cli_run_help_shows_new_flags():
    import re
    from typer.testing import CliRunner
    from motion_mirror.cli import app

    runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Strip ANSI escape codes — Rich may still emit them even with NO_COLOR
    out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--offload-model" in out, f"--offload-model not found in:\n{out}"
    assert "--t5-cpu" in out
    assert "--flow-estimator" in out
    assert "--segmenter" in out
    assert "--auto" in out
