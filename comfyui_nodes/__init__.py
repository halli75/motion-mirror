"""ComfyUI custom node entry point for Motion Mirror."""
from __future__ import annotations

from .nodes import MotionMirrorGenerate, MotionMirrorPoseExtract, MotionMirrorTrajectoryGen

NODE_CLASS_MAPPINGS = {
    "MotionMirrorPoseExtract": MotionMirrorPoseExtract,
    "MotionMirrorTrajectoryGen": MotionMirrorTrajectoryGen,
    "MotionMirrorGenerate": MotionMirrorGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MotionMirrorPoseExtract": "Motion Mirror Pose Extract",
    "MotionMirrorTrajectoryGen": "Motion Mirror Trajectory Gen",
    "MotionMirrorGenerate": "Motion Mirror Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
