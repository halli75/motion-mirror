"""Initial ComfyUI node scaffold for Motion Mirror."""
from __future__ import annotations

from pathlib import Path

from . import model_management


class MotionMirrorPoseExtract:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_video_path",)
    FUNCTION = "run"
    CATEGORY = "Motion Mirror"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_video_path": ("STRING", {"default": ""}),
            }
        }

    def run(self, motion_video_path: str) -> tuple[str]:
        path = Path(motion_video_path)
        if not path.exists():
            raise FileNotFoundError(f"Motion video not found: {path}")
        return (str(path),)


class MotionMirrorTrajectoryGen:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_video_path",)
    FUNCTION = "run"
    CATEGORY = "Motion Mirror"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_video_path": ("STRING", {"default": ""}),
            }
        }

    def run(self, motion_video_path: str) -> tuple[str]:
        path = Path(motion_video_path)
        if not path.exists():
            raise FileNotFoundError(f"Motion video not found: {path}")
        return (str(path),)


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
                        "wan-move-14b",
                        "wan-move-fast",
                        "wan-move-gguf",
                        "wan-1.3b-vace",
                        "wan-1.3b-concat-id",
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
            }
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
        )
