from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import get_args

from .config import BackendName, MotionMirrorConfig
from .extract.pose import extract_pose
from .extract.render_skeleton import render_skeleton_conditioning_artifacts
from .extract.segment import segment_subject
from .extract.trajectory import synthesize_trajectory
from .generate.models import GenerationRequest
from .generate.vace import generate_with_vace
from .hardware import auto_config
from .postprocess.audio import passthrough_audio
from .types import PoseSequence, TrajectoryMap


@dataclass(slots=True)
class PipelineRunResult:
    image_path: Path
    motion_video_path: Path
    output_path: Path
    segmentation_path: Path | None = None
    trajectory_path: Path | None = None
    conditioning_video_path: Path | None = None
    conditioning_mask_path: Path | None = None


class MotionMirrorPipeline:
    def __init__(self, config: MotionMirrorConfig | None = None) -> None:
        self.config = config or MotionMirrorConfig()

    def run(
        self,
        image_path: Path,
        motion_video_path: Path,
        *,
        pose: PoseSequence | None = None,
        trajectory: TrajectoryMap | None = None,
    ) -> PipelineRunResult:
        """Run the full motion transfer pipeline.

        ``pose`` and ``trajectory`` accept precomputed artifacts (e.g. from the
        ComfyUI PoseExtract / TrajectoryGen nodes) so the corresponding
        extraction stages are skipped.
        """
        cfg = self.config

        if cfg.backend == "auto":
            cfg = auto_config(cfg)

        # "auto" is resolved by auto_config above; every other BackendName is runnable.
        valid_backends = set(get_args(BackendName)) - {"auto"}
        if cfg.backend not in valid_backends:
            raise ValueError(
                f"Unknown backend {cfg.backend!r}. "
                f"Valid choices: {sorted(valid_backends)}."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Character image not found: {image_path}")
        if not motion_video_path.exists():
            raise FileNotFoundError(f"Motion video not found: {motion_video_path}")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        seg = segment_subject(image_path, cfg)
        if pose is None:
            pose = extract_pose(motion_video_path, cfg)

        conditioning_video_path: Path | None = None
        conditioning_mask_path: Path | None = None
        # "auto" is already resolved and validity checked above; every
        # non-mock backend is a VACE backend that needs the conditioning
        # pose + mask artifacts. Without this, the 14B backends would
        # silently receive no conditioning and fail input validation.
        if cfg.backend != "mock":
            conditioning_video_path = cfg.output_dir / "conditioning_pose.mp4"
            conditioning_mask_path = cfg.output_dir / "conditioning_mask.mp4"
            render_skeleton_conditioning_artifacts(
                pose_seq=pose,
                video_path=conditioning_video_path,
                mask_path=conditioning_mask_path,
                size=cfg.resolution_wh,
                num_frames=cfg.num_frames,
            )

        if trajectory is None:
            traj = synthesize_trajectory(
                pose,
                seg,
                motion_video_path,
                cfg,
            )
        else:
            traj = trajectory
        traj_path = cfg.output_dir / "trajectory.npz"
        traj.save(traj_path)

        gen_request = GenerationRequest(
            segmented_image_path=seg.rgba_path,
            trajectory_map_path=traj_path,
            output_path=cfg.output_dir / "generated.mp4",
            conditioning_video_path=conditioning_video_path,
            conditioning_mask_path=conditioning_mask_path,
            backend=cfg.backend,
            resolution=cfg.resolution,
            frames=cfg.num_frames,
            device=cfg.device,
        )

        gen = generate_with_vace(gen_request, cfg)

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
            conditioning_video_path=conditioning_video_path,
            conditioning_mask_path=conditioning_mask_path,
        )
