"""Tests for v0.2a config additions and pipeline wiring."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig


def test_config_default_backend():
    cfg = MotionMirrorConfig()
    assert cfg.backend == "wan-1.3b-vace"


def test_config_backends():
    for backend in (
        "auto",
        "wan-1.3b-vace",
        "mock",
    ):
        cfg = MotionMirrorConfig(backend=backend)
        assert cfg.backend == backend


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


@pytest.mark.parametrize(
    "backend", ["wan-1.3b-vace", "wan-14b-vace", "wan-14b-vace-gguf"]
)
def test_pipeline_vace_backend_threads_conditioning_artifacts(tmp_path, backend):
    """Every real VACE backend must receive conditioning artifacts from the
    pipeline (the branch is `!= "mock"` — a backend-name allowlist that forgot
    the 14B names would silently starve them of conditioning)."""
    from motion_mirror.pipeline import MotionMirrorPipeline

    img_path = tmp_path / "char.png"
    img_path.write_bytes(b"fake-image")
    vid_path = tmp_path / "motion.mp4"
    vid_path.write_bytes(b"fake-video")

    cfg = MotionMirrorConfig(
        backend=backend,
        resolution="64x32",
        num_frames=4,
        trajectory_density=16,
        device="cpu",
        project_root=tmp_path,
    )

    fake_seg = MagicMock()
    fake_seg.rgba_path = tmp_path / "segmented.png"
    fake_seg.rgba_path.touch()
    fake_pose = MagicMock()
    fake_traj = MagicMock()
    fake_traj.save = MagicMock()
    fake_gen = MagicMock()
    fake_gen.video_path = tmp_path / "generated.mp4"
    fake_gen.video_path.touch()
    final_path = tmp_path / "result.mp4"
    final_path.touch()

    def fake_render(**kwargs):
        kwargs["video_path"].touch()
        kwargs["mask_path"].touch()
        return kwargs["video_path"], kwargs["mask_path"]

    with (
        patch("motion_mirror.pipeline.segment_subject", return_value=fake_seg),
        patch("motion_mirror.pipeline.extract_pose", return_value=fake_pose),
        patch("motion_mirror.pipeline.synthesize_trajectory", return_value=fake_traj),
        patch("motion_mirror.pipeline.render_skeleton_conditioning_artifacts", side_effect=fake_render),
        patch("motion_mirror.pipeline.generate_with_vace", return_value=fake_gen) as gen_mock,
        patch("motion_mirror.pipeline.passthrough_audio", return_value=final_path),
    ):
        result = MotionMirrorPipeline(cfg).run(img_path, vid_path)

    req = gen_mock.call_args.args[0]
    assert req.conditioning_video_path == tmp_path / "outputs" / "conditioning_pose.mp4"
    assert req.conditioning_mask_path == tmp_path / "outputs" / "conditioning_mask.mp4"
    assert result.conditioning_video_path == req.conditioning_video_path
    assert result.conditioning_mask_path == req.conditioning_mask_path
    assert result.conditioning_video_path.exists()
    assert result.conditioning_mask_path.exists()


def test_pipeline_unknown_backend_raises(tmp_path):
    from motion_mirror.pipeline import MotionMirrorPipeline

    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    vid = tmp_path / "x.mp4"
    vid.write_bytes(b"fake")

    cfg = MotionMirrorConfig(backend="wan-1.3b-vace")
    object.__setattr__(cfg, "backend", "nonexistent-backend")
    with pytest.raises(ValueError, match="nonexistent-backend"):
        MotionMirrorPipeline(cfg).run(img, vid)


def test_pipeline_vace_backend_in_valid_set(tmp_path):
    from motion_mirror.pipeline import MotionMirrorPipeline

    img_path = tmp_path / "char.png"
    Image.new("RGB", (64, 64), (100, 120, 80)).save(str(img_path))

    vid_path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 64))
    rng = np.random.default_rng(2)
    for _ in range(3):
        writer.write(rng.integers(0, 200, (64, 64, 3), dtype=np.uint8))
    writer.release()

    cfg = MotionMirrorConfig(
        backend="wan-1.3b-vace",
        resolution="64x32",
        num_frames=5,
        trajectory_density=16,
        device="cpu",
        project_root=tmp_path,
    )
    # A real backend without weights fails downstream, but never with an
    # "unknown backend" ValueError from the dispatch guard.
    with pytest.raises(Exception) as exc_info:
        MotionMirrorPipeline(cfg).run(img_path, vid_path)
    assert "unknown" not in str(exc_info.value).lower() or exc_info.type is not ValueError


def test_cli_presets_list_shows_presets():
    from motion_mirror.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["presets", "--list"])
    assert result.exit_code == 0
    out = result.output.encode("ascii", errors="replace").decode()
    assert "default" in out
    assert "low-vram" in out
    assert "mock" in out


def test_cli_download_help_shows_groups():
    from motion_mirror.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "vace" in result.output or "wan-1.3b" in result.output
    assert "dwpose" in result.output


def test_cli_vace_download_spec():
    from motion_mirror.cli import _MODEL_GROUPS, _MODEL_SPECS

    assert _MODEL_GROUPS["vace"] == ["wan-1.3b-vace"]
    spec = _MODEL_SPECS["wan-1.3b-vace"]
    assert spec["repo_id"] == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert spec["cache_subdir"] == "wan-1.3b-vace"


def test_cli_run_help_shows_new_flags():
    from motion_mirror.cli import app
    from typer.testing import CliRunner

    runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "--offload-model" in out
    assert "--t5-cpu" in out
    assert "--flow-estimator" in out
    assert "--segmenter" in out
    assert "--auto" in out
    assert "wan-1.3b-vace" in out
