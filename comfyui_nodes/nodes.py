"""ComfyUI nodes for Motion Mirror.

PoseExtract and TrajectoryGen run the real CPU-capable extraction stages and
exchange artifacts as ``.npz`` file paths. Generate accepts those artifacts so
the three nodes compose as a graph, or runs the full pipeline when they are
omitted.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from . import model_management


def _artifact_dir() -> Path:
    """Directory for intermediate node artifacts.

    Uses ComfyUI's output directory when running inside ComfyUI, otherwise a
    temp directory (e.g. during tests).
    """
    try:
        import folder_paths  # type: ignore[import]

        base = Path(folder_paths.get_output_directory())
    except ImportError:
        base = Path(tempfile.gettempdir())
    out = base / "motion_mirror"
    out.mkdir(parents=True, exist_ok=True)
    return out


class MotionMirrorPoseExtract:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose_path",)
    FUNCTION = "run"
    CATEGORY = "Motion Mirror"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_video_path": ("STRING", {"default": ""}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "mock": ("BOOLEAN", {"default": False}),
            },
        }

    def run(
        self,
        motion_video_path: str,
        device: str,
        mock: bool = False,
    ) -> tuple[str]:
        path = Path(motion_video_path)
        if not path.exists():
            raise FileNotFoundError(f"Motion video not found: {path}")

        from motion_mirror.config import MotionMirrorConfig
        from motion_mirror.extract.pose import extract_pose

        model_management.maybe_throw_if_interrupted()
        config = MotionMirrorConfig(
            backend="mock" if mock else "wan-1.3b-vace",
            device=device,
        )
        pose = extract_pose(path, config)
        model_management.maybe_throw_if_interrupted()

        pose_path = _artifact_dir() / f"pose_{uuid.uuid4().hex}.npz"
        pose.save(pose_path)
        return (str(pose_path),)


class MotionMirrorTrajectoryGen:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trajectory_path",)
    FUNCTION = "run"
    CATEGORY = "Motion Mirror"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_path": ("STRING", {"default": ""}),
                "character_image_path": ("STRING", {"default": ""}),
                "motion_video_path": ("STRING", {"default": ""}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 241}),
                "density": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    def run(
        self,
        pose_path: str,
        character_image_path: str,
        motion_video_path: str,
        frames: int,
        density: int,
        device: str,
    ) -> tuple[str]:
        for label, raw in (
            ("Pose artifact", pose_path),
            ("Character image", character_image_path),
            ("Motion video", motion_video_path),
        ):
            if not Path(raw).exists():
                raise FileNotFoundError(f"{label} not found: {raw}")

        from motion_mirror.config import MotionMirrorConfig
        from motion_mirror.extract.segment import segment_subject
        from motion_mirror.extract.trajectory import synthesize_trajectory
        from motion_mirror.types import PoseSequence

        model_management.maybe_throw_if_interrupted()
        work_dir = _artifact_dir() / f"trajectory_{uuid.uuid4().hex}"
        config = MotionMirrorConfig(
            num_frames=int(frames),
            trajectory_density=int(density),
            device=device,
            project_root=work_dir,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)

        pose = PoseSequence.load(Path(pose_path))
        segmentation = segment_subject(Path(character_image_path), config)
        trajectory = synthesize_trajectory(
            pose,
            segmentation,
            Path(motion_video_path),
            config,
        )
        model_management.maybe_throw_if_interrupted()

        trajectory_path = work_dir / "trajectory.npz"
        trajectory.save(trajectory_path)
        return (str(trajectory_path),)


class MotionMirrorGenerate:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "run"
    CATEGORY = "Motion Mirror"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_image_path": ("STRING", {"default": ""}),
                "motion_video_path": ("STRING", {"default": ""}),
                "backend": (
                    [
                        "auto",
                        "wan-1.3b-vace",
                        "wan-14b-vace",
                        "wan-14b-vace-gguf",
                        "mock",
                    ],
                    {"default": "auto"},
                ),
                "resolution": (
                    ["832x480", "1280x720", "128x64"],
                    {"default": "832x480"},
                ),
                "frames": ("INT", {"default": 81, "min": 1, "max": 241}),
                "density": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "pose_path": ("STRING", {"default": ""}),
                "trajectory_path": ("STRING", {"default": ""}),
            },
        }

    def run(
        self,
        character_image_path: str,
        motion_video_path: str,
        backend: str,
        resolution: str,
        frames: int,
        density: int,
        device: str,
        pose_path: str = "",
        trajectory_path: str = "",
    ) -> tuple[str]:
        if not Path(character_image_path).exists():
            raise FileNotFoundError(f"Character image not found: {character_image_path}")
        if not Path(motion_video_path).exists():
            raise FileNotFoundError(f"Motion video not found: {motion_video_path}")
        return model_management.run_motion_mirror_generation(
            image_path=character_image_path,
            motion_video_path=motion_video_path,
            backend=backend,
            resolution=resolution,
            frames=frames,
            density=density,
            device=device,
            pose_path=pose_path,
            trajectory_path=trajectory_path,
        )
