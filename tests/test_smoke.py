from pathlib import Path

from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline


def test_pipeline_returns_default_output_path() -> None:
    config = MotionMirrorConfig(project_root=Path("/tmp/motion-mirror"))
    pipeline = MotionMirrorPipeline(config)

    result = pipeline.run(Path("image.png"), Path("motion.mp4"))

    assert result.output_path == Path("/tmp/motion-mirror/outputs/result.mp4")
