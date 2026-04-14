from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import warnings

from .config import MotionMirrorConfig
from .extract.pose import extract_pose
from .extract.segment import segment_subject
from .extract.trajectory import synthesize_trajectory
from .generate.controlnet import generate_with_controlnet
from .generate.models import GenerationRequest
from .generate.wan_move import generate_with_wan_move
from .hardware import auto_config
from .postprocess.audio import passthrough_audio


@dataclass(slots=True)
class PipelineRunResult:
    image_path: Path
    motion_video_path: Path
    output_path: Path
    segmentation_path: Path | None = None  # RGBA PNG from segment stage
    trajectory_path: Path | None = None    # .npz from trajectory stage


class MotionMirrorPipeline:
    def __init__(self, config: MotionMirrorConfig | None = None) -> None:
        self.config = config or MotionMirrorConfig()

    def run(
        self,
        image_path: Path,
        motion_video_path: Path,
    ) -> PipelineRunResult:
        """Run the full motion transfer pipeline.

        Stages (in order):
          1. Segmentation   — rembg background removal
          2. Pose extraction — DWPose-L keypoints from motion video
          3. Trajectory      — 3-layer dense track synthesis
          4. Generation      — Wan-Move-14B or ControlNet video synthesis
          5. Audio           — passthrough audio from source video

        Parameters
        ----------
        image_path:
            Path to the character image (PNG/JPG/JPEG/WEBP).
        motion_video_path:
            Path to the reference motion video (MP4/MOV/AVI/MKV).

        Returns
        -------
        PipelineRunResult
            Contains output_path plus debug paths for intermediate artefacts.

        Raises
        ------
        FileNotFoundError
            If either input file does not exist.
        ValueError
            If backend is unknown.
        UnsupportedImageError
            If the character image format is not supported.
        UnsupportedVideoError
            If the motion video format is not supported.
        VideoDecodeError
            If the video cannot be decoded.
        NoPoseDetectedError
            If no person is found in the reference video.
        MultiplePeopleDetectedError
            If more than one person is found in the reference video.
        SmallSubjectError
            If the detected person is too small in the reference video.
        """
        cfg = self.config

        # ── Resolve 'auto' backend from available VRAM ───────────────────────
        if cfg.backend == "auto":
            cfg = auto_config(cfg)

        # ── Deprecation: 'controlnet' → 'wan-1.3b-vace' ─────────────────────
        if cfg.backend == "controlnet":
            warnings.warn(
                "backend='controlnet' is deprecated — use 'wan-1.3b-vace' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(cfg, "backend", "wan-1.3b-vace")

        # ── Input validation ─────────────────────────────────────────────────
        _VALID_BACKENDS = {"wan-move-14b", "wan-move-fast", "wan-1.3b-vace", "mock"}
        if cfg.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {cfg.backend!r}. "
                f"Valid choices: {sorted(_VALID_BACKENDS)}."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Character image not found: {image_path}")
        if not motion_video_path.exists():
            raise FileNotFoundError(f"Motion video not found: {motion_video_path}")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: Segmentation ────────────────────────────────────────────
        seg = segment_subject(image_path, cfg)

        # ── Stage 2: Pose extraction ─────────────────────────────────────────
        pose = extract_pose(motion_video_path, cfg)

        # ── Stage 3: Trajectory synthesis ───────────────────────────────────
        traj = synthesize_trajectory(pose, seg, motion_video_path, cfg)
        traj_path = cfg.output_dir / "trajectory.npz"
        traj.save(traj_path)

        # ── Stage 4: Generation ──────────────────────────────────────────────
        gen_request = GenerationRequest(
            segmented_image_path=seg.rgba_path,
            trajectory_map_path=traj_path,
            output_path=cfg.output_dir / "generated.mp4",
            backend=cfg.backend,
            resolution=cfg.resolution,
            frames=cfg.num_frames,
            device=cfg.device,
        )

        if cfg.backend in ("wan-move-14b", "wan-move-fast", "mock"):
            gen = generate_with_wan_move(gen_request, cfg)
        else:  # "wan-1.3b-vace"
            gen = generate_with_controlnet(gen_request, cfg)

        # ── Stage 5: Audio passthrough ───────────────────────────────────────
        final_path = passthrough_audio(
            source_video_path=motion_video_path,
            generated_video_path=gen.video_path,
            output_path=cfg.output_dir / "result.mp4",
        )

        return PipelineRunResult(
            image_path=image_path,
            motion_video_path=motion_video_path,
            output_path=final_path,
            segmentation_path=seg.rgba_path,
            trajectory_path=traj_path,
        )
