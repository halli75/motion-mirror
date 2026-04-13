from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MotionMirrorConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    trajectory_density: int = 512
    output_dir_name: str = "outputs"

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_dir_name
