"""Gradio web UI for Motion Mirror."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import MotionMirrorConfig


def _run_pipeline(
    img_path: str | None,
    vid_path: str | None,
    backend: str,
    resolution: str,
    frames: int | float,
    density: int | float,
    device: str,
) -> tuple[str | None, str]:
    """Core pipeline execution logic, extracted for testability.

    Returns (output_video_path_or_None, status_message).
    """
    from ..config import MotionMirrorConfig
    from ..pipeline import MotionMirrorPipeline

    if img_path is None or vid_path is None:
        return None, "Error: provide both a character image and a motion video."

    run_cfg = MotionMirrorConfig(
        backend=backend,
        resolution=resolution,
        num_frames=int(frames),
        trajectory_density=int(density),
        device=device,
    )
    try:
        result = MotionMirrorPipeline(run_cfg).run(
            Path(img_path), Path(vid_path)
        )
        return str(result.output_path), f"Done. Output: {result.output_path}"
    except FileNotFoundError as exc:
        return None, f"Error: {exc}"
    except NotImplementedError as exc:
        return None, f"Not implemented: {exc}"
    except Exception as exc:
        return None, f"Pipeline error: {exc}"


def create_app(config: "MotionMirrorConfig | None" = None):
    """Build and return the Gradio Blocks demo.

    Parameters
    ----------
    config:
        Optional ``MotionMirrorConfig`` to use as defaults for the UI controls.
        If omitted a fresh default config is used.
    """
    import gradio as gr

    from ..config import MotionMirrorConfig

    cfg_defaults = config or MotionMirrorConfig()

    with gr.Blocks(title="Motion Mirror") as demo:
        gr.Markdown(
            "# Motion Mirror\n"
            "Transfer motion from a reference video onto a character image. "
            "Local-first — all inference runs on your machine."
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                char_image = gr.Image(
                    type="filepath",
                    label="Character Image",
                    sources=["upload"],
                )
                motion_video = gr.Video(
                    label="Reference Motion Video",
                    sources=["upload"],
                )

            # ── Right column: output ─────────────────────────────────────────
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Generated Animation",
                    interactive=False,
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

        # ── Advanced settings ────────────────────────────────────────────────
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                backend_dd = gr.Dropdown(
                    choices=["auto", "wan-move-14b", "wan-move-fast", "wan-1.3b-vace", "mock"],
                    value=cfg_defaults.backend,
                    label="Backend",
                )
                resolution_dd = gr.Dropdown(
                    choices=["832x480", "1280x720", "128x64"],
                    value=cfg_defaults.resolution,
                    label="Resolution",
                )
                device_dd = gr.Dropdown(
                    choices=["cuda", "cpu"],
                    value=cfg_defaults.device,
                    label="Device",
                )
            with gr.Row():
                frames_sl = gr.Slider(
                    minimum=4,
                    maximum=121,
                    step=4,
                    value=cfg_defaults.num_frames,
                    label="Frames",
                )
                density_sl = gr.Slider(
                    minimum=32,
                    maximum=1024,
                    step=32,
                    value=cfg_defaults.trajectory_density,
                    label="Trajectory Density",
                )

        run_btn = gr.Button("Generate", variant="primary")

        run_btn.click(
            fn=_run_pipeline,
            inputs=[
                char_image,
                motion_video,
                backend_dd,
                resolution_dd,
                frames_sl,
                density_sl,
                device_dd,
            ],
            outputs=[output_video, status_box],
        )

    demo.queue()
    return demo
