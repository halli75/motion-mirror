"""Basic smoke tests — import and instantiation only, no file I/O."""
from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline
from motion_mirror.pipeline import PipelineRunResult


def test_pipeline_instantiates_with_defaults() -> None:
    pipeline = MotionMirrorPipeline()
    assert pipeline.config.backend == "wan-move-14b"
    assert pipeline.config.trajectory_density == 512


def test_pipeline_instantiates_with_custom_config() -> None:
    config = MotionMirrorConfig(backend="mock", device="cpu", trajectory_density=128)
    pipeline = MotionMirrorPipeline(config)
    assert pipeline.config.backend == "mock"
    assert pipeline.config.trajectory_density == 128


def test_pipeline_run_result_fields() -> None:
    """PipelineRunResult has all expected fields."""
    from pathlib import Path
    result = PipelineRunResult(
        image_path=Path("img.png"),
        motion_video_path=Path("motion.mp4"),
        output_path=Path("outputs/result.mp4"),
        segmentation_path=Path("outputs/segmented.png"),
        trajectory_path=Path("outputs/trajectory.npz"),
    )
    assert result.output_path == Path("outputs/result.mp4")
    assert result.segmentation_path == Path("outputs/segmented.png")
    assert result.trajectory_path == Path("outputs/trajectory.npz")


def test_pipeline_run_result_optional_fields_default_none() -> None:
    from pathlib import Path
    result = PipelineRunResult(
        image_path=Path("img.png"),
        motion_video_path=Path("motion.mp4"),
        output_path=Path("out.mp4"),
    )
    assert result.segmentation_path is None
    assert result.trajectory_path is None
