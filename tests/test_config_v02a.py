"""Tests for v0.2a config additions and pipeline wiring."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from motion_mirror.config import MotionMirrorConfig


def test_config_default_backend():
    cfg = MotionMirrorConfig()
    assert cfg.backend == "wan-move-14b"


def test_config_new_backends():
    for backend in ("auto", "wan-move-fast", "wan-1.3b-vace", "controlnet", "mock"):
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


def test_pipeline_controlnet_alias_warns(tmp_path):
    from motion_mirror.pipeline import MotionMirrorPipeline

    img_path = tmp_path / "char.png"
    img_path.write_bytes(b"fake-image")
    vid_path = tmp_path / "motion.mp4"
    vid_path.write_bytes(b"fake-video")

    run_cfg = MotionMirrorConfig(
        backend="controlnet",
        resolution="64x32",
        num_frames=2,
        trajectory_density=16,
        device="cpu",
        project_root=tmp_path,
    )

    fake_seg = MagicMock()
    fake_seg.rgba_path = tmp_path / "seg.png"
    fake_seg.rgba_path.touch()
    fake_pose = MagicMock()
    fake_traj = MagicMock()
    fake_traj.save = MagicMock()
    fake_gen = MagicMock()
    fake_gen.video_path = tmp_path / "gen.mp4"
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
        patch("motion_mirror.pipeline.render_skeleton_conditioning_artifacts", side_effect=fake_render),
        patch("motion_mirror.pipeline.synthesize_trajectory", return_value=fake_traj),
        patch("motion_mirror.pipeline.generate_with_controlnet", return_value=fake_gen),
        patch("motion_mirror.pipeline.passthrough_audio", return_value=final_path),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MotionMirrorPipeline(run_cfg).run(img_path, vid_path)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings
    assert "controlnet" in str(dep_warnings[0].message).lower()
    assert result.output_path.exists()


def test_pipeline_vace_backend_threads_conditioning_artifacts(tmp_path):
    from motion_mirror.pipeline import MotionMirrorPipeline

    img_path = tmp_path / "char.png"
    img_path.write_bytes(b"fake-image")
    vid_path = tmp_path / "motion.mp4"
    vid_path.write_bytes(b"fake-video")

    cfg = MotionMirrorConfig(
        backend="wan-1.3b-vace",
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
        patch("motion_mirror.pipeline.generate_with_controlnet", return_value=fake_gen) as gen_mock,
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

    cfg = MotionMirrorConfig(backend="wan-move-14b")
    object.__setattr__(cfg, "backend", "nonexistent-backend")
    with pytest.raises(ValueError, match="nonexistent-backend"):
        MotionMirrorPipeline(cfg).run(img, vid)


def test_pipeline_new_backends_in_valid_set(tmp_path):
    from motion_mirror.pipeline import MotionMirrorPipeline

    img_path = tmp_path / "char.png"
    Image.new("RGB", (64, 64), (100, 120, 80)).save(str(img_path))

    vid_path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 64))
    rng = np.random.default_rng(2)
    for _ in range(3):
        writer.write(rng.integers(0, 200, (64, 64, 3), dtype=np.uint8))
    writer.release()

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
        assert "unknown" not in str(exc_info.value).lower() or exc_info.type is not ValueError


def test_cli_presets_list_shows_new_presets():
    from motion_mirror.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["presets", "--list"])
    assert result.exit_code == 0
    out = result.output.encode("ascii", errors="replace").decode()
    assert "low-vram" in out
    assert "fast" in out


def test_cli_download_help_shows_new_groups():
    from motion_mirror.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "light" in result.output or "wan-1.3b" in result.output


def test_cli_fast_download_spec_uses_lightx2v_assets():
    from motion_mirror.cli import _MODEL_SPECS

    spec = _MODEL_SPECS["wan-move-fast"]
    assert spec["cache_subdir"] == "wan-move-fast"
    assert "wan_i2v_distill_4step_cfg_4090.json" in spec["required_paths"]
    assert "sources" in spec
    repo_ids = {source["repo_id"] for source in spec["sources"]}
    assert "lightx2v/Wan2.1-Distill-Models" in repo_ids
    assert "Wan-AI/Wan2.1-I2V-14B-720P" in repo_ids


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
