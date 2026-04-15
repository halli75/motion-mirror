"""Audio passthrough post-processing.

Muxes the audio stream from the reference motion video into the generated
video using ffmpeg (via static-ffmpeg + ffmpeg-python).

If the source video has no audio stream, the generated video is returned
as-is without re-encoding.

static_ffmpeg.add_paths() is called at module load so that the bundled
ffmpeg binary is on PATH before any ffmpeg-python calls.
"""
from __future__ import annotations

from pathlib import Path

try:
    import static_ffmpeg  # type: ignore[import]
    static_ffmpeg.add_paths()
except ImportError:
    pass  # ffmpeg-python will fall back to system ffmpeg if present

import ffmpeg  # type: ignore[import]


def passthrough_audio(
    source_video_path: Path,
    generated_video_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Copy the audio track from *source_video_path* into *generated_video_path*.

    Parameters
    ----------
    source_video_path:
        The original reference motion video (audio donor).
    generated_video_path:
        The model-generated silent video.
    output_path:
        Where to write the muxed result.  Defaults to
        ``generated_video_path.parent / "final.mp4"``.

    Returns
    -------
    Path
        Path to the output video (with or without audio, depending on source).

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    RuntimeError
        If ffmpeg fails during muxing.
    """
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not generated_video_path.exists():
        raise FileNotFoundError(f"Generated video not found: {generated_video_path}")

    # Probe source for audio streams
    try:
        probe = ffmpeg.probe(str(source_video_path))
    except ffmpeg.Error as exc:
        raise RuntimeError(
            f"ffmpeg probe failed on {source_video_path}: "
            f"{exc.stderr.decode(errors='replace') if exc.stderr else exc}"
        ) from exc

    has_audio = any(
        s.get("codec_type") == "audio" for s in probe.get("streams", [])
    )

    if not has_audio:
        # No audio to copy — return generated video unchanged
        return generated_video_path

    out = output_path or (generated_video_path.parent / "final.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        video_in = ffmpeg.input(str(generated_video_path))
        audio_in = ffmpeg.input(str(source_video_path)).audio
        (
            ffmpeg
            .output(
                video_in,
                audio_in,
                str(out),
                vcodec="copy",
                acodec="aac",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(
            f"ffmpeg mux failed: {exc.stderr.decode(errors='replace') if exc.stderr else exc}"
        ) from exc

    return out
