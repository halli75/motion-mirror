from dataclasses import dataclass
from pathlib import Path

from .config import MotionMirrorConfig


@dataclass(slots=True)
class PipelineRunResult:
    image_path: Path
    motion_video_path: Path
    output_path: Path


class MotionMirrorPipeline:
    def __init__(self, config: MotionMirrorConfig | None = None) -> None:
        self.config = config or MotionMirrorConfig()

    def run(self, image_path: Path, motion_video_path: Path) -> PipelineRunResult:
        output_path = self.config.output_dir / "result.mp4"
        return PipelineRunResult(
            image_path=image_path,
            motion_video_path=motion_video_path,
            output_path=output_path,
        )
