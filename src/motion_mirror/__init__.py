from .config import MotionMirrorConfig
from .exceptions import (
    HardwareError,
    InputError,
    InsufficientVRAMError,
    MotionMirrorError,
    MultipleCharactersError,
    MultiplePeopleDetectedError,
    NoPoseDetectedError,
    SmallSubjectError,
    SmallSubjectWarning,
    SubjectError,
    UnsupportedImageError,
    UnsupportedVideoError,
    VideoDecodeError,
)
from .hardware import GPUInfo, auto_config, get_gpu_info, recommend_backend
from .pipeline import MotionMirrorPipeline
from .types import (
    GenerationResult,
    PoseSequence,
    SegmentationResult,
    TrajectoryMap,
)

__all__ = [
    "MotionMirrorConfig",
    "MotionMirrorPipeline",
    "SegmentationResult",
    "PoseSequence",
    "TrajectoryMap",
    "GenerationResult",
    # Hardware
    "GPUInfo",
    "get_gpu_info",
    "recommend_backend",
    "auto_config",
    # Exceptions
    "MotionMirrorError",
    "InputError",
    "UnsupportedImageError",
    "UnsupportedVideoError",
    "VideoDecodeError",
    "NoPoseDetectedError",
    "MultiplePeopleDetectedError",
    "SubjectError",
    "SmallSubjectWarning",
    "SmallSubjectError",
    "MultipleCharactersError",
    "HardwareError",
    "InsufficientVRAMError",
]
