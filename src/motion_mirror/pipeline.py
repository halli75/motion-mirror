from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import warnings

from .config import MotionMirrorConfig
from .extract.pose import extract_pose
from .extract.render_skeleton import render_skeleton_conditioning_artifacts
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
    ) -> PipelineRunResult:
        """Run the full motion transfer pipeline."""
        cfg = self.config

        if cfg.backend == "auto":
            cfg = auto_config(cfg)

        if cfg.backend == "controlnet":
            warnings.warn(
                "backend='controlnet' is deprecated - use 'wan-1.3b-vace' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cfg = dataclasses.replace(cfg, backend="wan-1.3b-vace")

        valid_backends = {"wan-move-14b", "wan-move-fast", "wan-1.3b-vace", "mock"}
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
        pose = extract_pose(motion_video_path, cfg)

        conditioning_video_path: Path | None = None
        conditioning_mask_path: Path | None = None
        if cfg.backend == "wan-1.3b-vace":
            conditioning_video_path = cfg.output_dir / "conditioning_pose.mp4"
            conditioning_mask_path = cfg.output_dir / "conditioning_mask.mp4"
            render_skeleton_conditioning_artifacts(
                pose_seq=pose,
                video_path=conditioning_video_path,
                mask_path=conditioning_mask_path,
                size=cfg.resolution_wh,
                num_frames=cfg.num_frames,
            )

        traj = synthesize_trajectory(pose, seg, motion_video_path, cfg)
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

        if cfg.backend in ("wan-move-14b", "wan-move-fast", "mock"):
            gen = generate_with_wan_move(gen_request, cfg)
        else:
            gen = generate_with_controlnet(gen_request, cfg)

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
