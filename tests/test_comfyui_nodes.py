from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


def _write_sample_video(path: Path, frames: int = 5, size: int = 64) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (size, size)
    )
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 200, (size, size, 3), dtype=np.uint8))
    writer.release()


def test_comfyui_node_mappings_include_v02b_nodes():
    import comfyui_nodes

    assert "MotionMirrorPoseExtract" in comfyui_nodes.NODE_CLASS_MAPPINGS
    assert "MotionMirrorTrajectoryGen" in comfyui_nodes.NODE_CLASS_MAPPINGS
    assert "MotionMirrorGenerate" in comfyui_nodes.NODE_CLASS_MAPPINGS


def test_motion_mirror_generate_routes_through_model_management(tmp_path, monkeypatch):
    from comfyui_nodes.nodes import MotionMirrorGenerate

    image = tmp_path / "char.png"
    motion = tmp_path / "motion.mp4"
    image.write_bytes(b"image")
    motion.write_bytes(b"video")
    calls: dict[str, object] = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return (str(tmp_path / "result.mp4"),)

    monkeypatch.setattr(
        "comfyui_nodes.nodes.model_management.run_motion_mirror_generation",
        fake_generate,
    )

    result = MotionMirrorGenerate().run(
        str(image),
        str(motion),
        "wan-1.3b-concat-id",
        "832x480",
        81,
        512,
        "cuda",
        pose_path="pose.npz",
        trajectory_path="trajectory.npz",
    )

    assert result == (str(tmp_path / "result.mp4"),)
    assert calls["backend"] == "wan-1.3b-concat-id"
    assert calls["image_path"] == str(image)
    assert calls["motion_video_path"] == str(motion)
    assert calls["pose_path"] == "pose.npz"
    assert calls["trajectory_path"] == "trajectory.npz"


def test_pose_extract_node_writes_loadable_pose_artifact(tmp_path, monkeypatch):
    """PoseExtract must run extract_pose and persist a loadable PoseSequence."""
    from comfyui_nodes import nodes
    from comfyui_nodes.nodes import MotionMirrorPoseExtract
    from motion_mirror.types import PoseSequence

    monkeypatch.setattr(nodes, "_artifact_dir", lambda: tmp_path)

    motion = tmp_path / "motion.mp4"
    _write_sample_video(motion)

    (pose_path,) = MotionMirrorPoseExtract().run(str(motion), "cpu", mock=True)

    loaded = PoseSequence.load(Path(pose_path))
    assert loaded.keypoints.shape == (5, 133, 3)
    assert loaded.frame_size == (64, 64)
    assert loaded.fps > 0


def test_trajectory_gen_node_runs_extraction_and_saves_artifact(tmp_path, monkeypatch):
    """TrajectoryGen must load the pose artifact, segment, synthesize, and save."""
    from comfyui_nodes import nodes
    from comfyui_nodes.nodes import MotionMirrorTrajectoryGen
    from motion_mirror.types import PoseSequence, SegmentationResult, TrajectoryMap

    monkeypatch.setattr(nodes, "_artifact_dir", lambda: tmp_path)

    motion = tmp_path / "motion.mp4"
    _write_sample_video(motion)
    image = tmp_path / "char.png"
    image.write_bytes(b"png")

    pose = PoseSequence(
        source_video_path=motion,
        keypoints=np.zeros((5, 133, 3), dtype=np.float32),
        frame_size=(64, 64),
        fps=24.0,
    )
    pose_path = tmp_path / "pose.npz"
    pose.save(pose_path)

    seen: dict[str, object] = {}

    def fake_segment(image_path, config):
        seen["segment_image"] = image_path
        return SegmentationResult(
            source_image_path=image_path,
            rgba_path=tmp_path / "segmented.png",
            mask=np.full((64, 64), 255, dtype=np.uint8),
            rgba=np.zeros((64, 64, 4), dtype=np.uint8),
        )

    def fake_synthesize(pose_seq, segmentation, motion_video_path, config):
        seen["pose_frames"] = pose_seq.keypoints.shape[0]
        seen["motion_video"] = motion_video_path
        seen["density"] = config.trajectory_density
        return TrajectoryMap(
            tracks=np.zeros((3, 16, 2), dtype=np.float32),
            flow_fields=np.zeros((2, 8, 8, 2), dtype=np.float32),
            density=16,
            frame_size=(64, 64),
        )

    monkeypatch.setattr(
        "motion_mirror.extract.segment.segment_subject", fake_segment
    )
    monkeypatch.setattr(
        "motion_mirror.extract.trajectory.synthesize_trajectory", fake_synthesize
    )

    (trajectory_path,) = MotionMirrorTrajectoryGen().run(
        str(pose_path), str(image), str(motion), 3, 16, "cpu"
    )

    assert seen["segment_image"] == image
    assert seen["pose_frames"] == 5
    assert seen["motion_video"] == motion
    assert seen["density"] == 16
    loaded = TrajectoryMap.load(Path(trajectory_path))
    assert loaded.density == 16
    assert loaded.tracks.shape == (3, 16, 2)


def test_model_management_passes_precomputed_artifacts_to_pipeline(tmp_path, monkeypatch):
    """run_motion_mirror_generation must load npz artifacts and forward them."""
    import comfyui_nodes.model_management as model_management
    from motion_mirror.types import PoseSequence, TrajectoryMap

    monkeypatch.setattr(model_management, "throw_if_interrupted", lambda: None)

    pose_path = tmp_path / "pose.npz"
    PoseSequence(
        source_video_path=tmp_path / "motion.mp4",
        keypoints=np.zeros((4, 133, 3), dtype=np.float32),
        frame_size=(64, 64),
        fps=24.0,
    ).save(pose_path)

    trajectory_path = tmp_path / "trajectory.npz"
    TrajectoryMap(
        tracks=np.zeros((4, 8, 2), dtype=np.float32),
        flow_fields=np.zeros((3, 8, 8, 2), dtype=np.float32),
        density=8,
        frame_size=(64, 64),
    ).save(trajectory_path)

    seen: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, config):
            seen["backend"] = config.backend

        def run(self, image_path, motion_video_path, *, pose=None, trajectory=None):
            seen["pose"] = pose
            seen["trajectory"] = trajectory
            return types.SimpleNamespace(output_path=tmp_path / "result.mp4")

    import motion_mirror

    monkeypatch.setattr(motion_mirror, "MotionMirrorPipeline", FakePipeline)

    result = model_management.run_motion_mirror_generation(
        image_path=str(tmp_path / "char.png"),
        motion_video_path=str(tmp_path / "motion.mp4"),
        backend="mock",
        resolution="64x32",
        frames=4,
        density=8,
        device="cpu",
        pose_path=str(pose_path),
        trajectory_path=str(trajectory_path),
    )

    assert result == (str(tmp_path / "result.mp4"),)
    assert seen["backend"] == "mock"
    assert seen["pose"].keypoints.shape == (4, 133, 3)
    assert seen["trajectory"].density == 8


def test_pipeline_skips_extraction_when_artifacts_precomputed(tmp_path, monkeypatch):
    """Pipeline.run(pose=..., trajectory=...) must not re-run extraction stages."""
    from PIL import Image

    import motion_mirror.pipeline as pipeline_mod
    from motion_mirror.config import MotionMirrorConfig
    from motion_mirror.types import PoseSequence, TrajectoryMap

    image = tmp_path / "char.png"
    Image.new("RGB", (64, 64), (180, 120, 80)).save(str(image))
    motion = tmp_path / "motion.mp4"
    _write_sample_video(motion)

    def fail_extract_pose(*args, **kwargs):
        raise AssertionError("extract_pose must not run with a precomputed pose")

    def fail_synthesize(*args, **kwargs):
        raise AssertionError(
            "synthesize_trajectory must not run with a precomputed trajectory"
        )

    monkeypatch.setattr(pipeline_mod, "extract_pose", fail_extract_pose)
    monkeypatch.setattr(pipeline_mod, "synthesize_trajectory", fail_synthesize)

    pose = PoseSequence(
        source_video_path=motion,
        keypoints=np.zeros((3, 133, 3), dtype=np.float32),
        frame_size=(64, 64),
        fps=24.0,
    )
    trajectory = TrajectoryMap(
        tracks=np.zeros((3, 16, 2), dtype=np.float32),
        flow_fields=np.zeros((2, 8, 8, 2), dtype=np.float32),
        density=16,
        frame_size=(64, 64),
    )

    config = MotionMirrorConfig(
        backend="mock",
        resolution="64x32",
        num_frames=3,
        trajectory_density=16,
        device="cpu",
        project_root=tmp_path,
    )
    result = pipeline_mod.MotionMirrorPipeline(config).run(
        image, motion, pose=pose, trajectory=trajectory
    )

    assert result.output_path.exists()
    assert result.trajectory_path is not None and result.trajectory_path.exists()


def test_model_management_uses_comfy_interrupt_hook(monkeypatch):
    import comfyui_nodes.model_management as model_management

    interrupted = {"count": 0}

    def fake_interrupt_hook():
        interrupted["count"] += 1

    comfy_mod = types.ModuleType("comfy")
    comfy_mm_mod = types.ModuleType("comfy.model_management")
    comfy_mm_mod.throw_exception_if_processing_interrupted = fake_interrupt_hook
    comfy_mod.model_management = comfy_mm_mod
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_mm_mod)

    model_management.throw_if_interrupted()

    assert interrupted["count"] == 1
