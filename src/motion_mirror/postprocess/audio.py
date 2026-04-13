from pathlib import Path


def passthrough_audio(source_video_path: Path, generated_video_path: Path) -> tuple[Path, Path]:
    return source_video_path, generated_video_path
