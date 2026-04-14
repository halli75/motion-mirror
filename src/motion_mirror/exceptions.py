"""Motion Mirror custom exception hierarchy.

All exceptions inherit from ``MotionMirrorError`` so callers can catch the
whole family with a single ``except MotionMirrorError`` clause.

Hierarchy
---------
MotionMirrorError
  InputError
    UnsupportedImageError
    UnsupportedVideoError
    VideoDecodeError
  PoseError
    NoPoseDetectedError
    MultiplePeopleDetectedError
  SubjectError
    SmallSubjectWarning   (Warning, not Exception)
    SmallSubjectError
    MultipleCharactersError
"""
from __future__ import annotations


# ── Base ──────────────────────────────────────────────────────────────────────

class MotionMirrorError(Exception):
    """Base class for all Motion Mirror exceptions."""


# ── Input validation ──────────────────────────────────────────────────────────

class InputError(MotionMirrorError):
    """Raised when an input file fails validation before processing begins."""


class UnsupportedImageError(InputError):
    """Raised when the character image format is not supported."""


class UnsupportedVideoError(InputError):
    """Raised when the reference video format is not supported."""


class VideoDecodeError(InputError):
    """Raised when ffmpeg or OpenCV cannot decode the reference video.

    Attributes
    ----------
    ffmpeg_output:
        Raw stderr from ffmpeg (if available) to help the user diagnose the
        root cause.
    """

    def __init__(self, message: str, ffmpeg_output: str = "") -> None:
        super().__init__(message)
        self.ffmpeg_output = ffmpeg_output


# ── Pose detection ────────────────────────────────────────────────────────────

class PoseError(MotionMirrorError):
    """Raised when DWPose cannot produce a usable pose for generation."""


class NoPoseDetectedError(PoseError):
    """Raised when DWPose finds no person in the reference video.

    Example
    -------
    ::

        raise NoPoseDetectedError(
            "No person detected in reference video. "
            "Ensure a person is clearly visible in the frame."
        )
    """


class MultiplePeopleDetectedError(PoseError):
    """Raised when DWPose finds more than one person in the reference video.

    Motion Mirror v0.1 supports single-person transfer only.  Use
    ``--person-index N`` to select by bounding-box area (0 = largest).

    Attributes
    ----------
    count:
        Number of people detected.
    """

    def __init__(self, message: str, count: int = 0) -> None:
        super().__init__(message)
        self.count = count


# ── Subject size / character validation ───────────────────────────────────────

class SubjectError(MotionMirrorError):
    """Raised for subject-size or character-count problems."""


class SmallSubjectWarning(UserWarning):
    """Issued (not raised) when the detected person occupies 5–10 % of the frame.

    Use ``warnings.warn(SmallSubjectWarning(...))`` rather than ``raise``.
    The pipeline continues but output quality may be degraded.
    """


class SmallSubjectError(SubjectError):
    """Raised when the detected person occupies less than 5 % of the frame area.

    Attributes
    ----------
    bbox_fraction:
        Fraction of frame area occupied by the detected person (0–1).
    """

    def __init__(self, message: str, bbox_fraction: float = 0.0) -> None:
        super().__init__(message)
        self.bbox_fraction = bbox_fraction


class MultipleCharactersError(SubjectError):
    """Raised when DWPose detects more than one person in the character image.

    The character image should contain exactly one person.  Crop the image to
    a single character before passing it to the pipeline.

    Attributes
    ----------
    count:
        Number of people detected in the character image.
    """

    def __init__(self, message: str, count: int = 0) -> None:
        super().__init__(message)
        self.count = count
