from pathlib import Path

import typer

from .config import MotionMirrorConfig
from .pipeline import MotionMirrorPipeline
from .ui.app import create_app

app = typer.Typer(help="Motion Mirror command line interface.")


@app.command()
def run(image: Path, motion: Path) -> None:
    pipeline = MotionMirrorPipeline(MotionMirrorConfig())
    result = pipeline.run(image, motion)
    typer.echo(f"Planned output: {result.output_path}")


@app.command()
def ui() -> None:
    demo = create_app()
    demo.launch()
