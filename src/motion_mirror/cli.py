"""Motion Mirror command-line interface.

Commands
--------
run         Run the motion transfer pipeline on an image + video.
download    Download model weights to the local cache.
presets     List available generation presets.
benchmark   Print GPU diagnostics and VRAM info.
ui          Launch the Gradio web UI.
"""
from __future__ import annotations

import shutil
import sys
import tomllib
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import MotionMirrorConfig
from .pipeline import MotionMirrorPipeline

app = typer.Typer(
    help="Motion Mirror — local-first motion transfer.",
    no_args_is_help=True,
)
console = Console()

# ── Preset helpers ─────────────────────────────────────────────────────────────

_PRESETS_DIR = Path(__file__).parent.parent.parent / "presets"


def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.toml"
    if not path.exists():
        available = [p.stem for p in sorted(_PRESETS_DIR.glob("*.toml"))]
        raise typer.BadParameter(
            f"Preset {name!r} not found. Available: {available}"
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))["preset"]


# ── Model download specs ───────────────────────────────────────────────────────

_MODEL_SPECS: dict[str, dict] = {
    "dwpose-pose": {
        "repo_id": "yzd-v/DWPose",
        "filename": "dw-ll_ucoco_384.onnx",
        "expected_bytes": 134_000_000,
        "cache_subdir": "dwpose",
        "label": "DWPose wholebody pose model",
    },
    "dwpose-det": {
        "repo_id": "yzd-v/DWPose",
        "filename": "yolox_l.onnx",
        "expected_bytes": 217_000_000,
        "cache_subdir": "dwpose",
        "label": "DWPose YOLOX detector",
    },
    "wan-move": {
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "filename": None,  # None = full snapshot_download
        "expected_bytes": 28_000_000_000,
        "cache_subdir": "wan-move",
        "label": "Wan2.1-I2V-14B-720P (diffusers format, ~28 GB)",
    },
}

_MODEL_GROUPS = {
    "dwpose": ["dwpose-pose", "dwpose-det"],
    "wan-move": ["wan-move"],
    "all": list(_MODEL_SPECS.keys()),
}


# ── run ───────────────────────────────────────────────────────────────────────


@app.command()
def run(
    image: Path = typer.Argument(..., help="Character image path (PNG/JPG/WEBP)."),
    motion: Path = typer.Argument(..., help="Reference motion video path (MP4/MOV/AVI/MKV)."),
    backend: Optional[str] = typer.Option(None, help="Generation backend: wan-move-14b | controlnet | mock."),
    resolution: Optional[str] = typer.Option(None, help="Output resolution WxH, e.g. 832x480."),
    frames: Optional[int] = typer.Option(None, help="Number of output frames."),
    density: Optional[int] = typer.Option(None, help="Trajectory density (512 = default, 1024 = HQ)."),
    device: Optional[str] = typer.Option(None, help="Compute device: cuda | cpu."),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: ./outputs)."),
    preset: Optional[str] = typer.Option(None, help="Load settings from a preset name (default | hq | mock)."),
) -> None:
    """Run the full motion transfer pipeline."""
    # Start from preset defaults, then apply explicit CLI overrides
    cfg_kwargs: dict = {}
    if preset:
        p = _load_preset(preset)
        cfg_kwargs["backend"] = p.get("backend", "wan-move-14b")
        cfg_kwargs["resolution"] = p.get("resolution", "832x480")
        cfg_kwargs["num_frames"] = p.get("num_frames", 81)
        cfg_kwargs["trajectory_density"] = p.get("trajectory_density", 512)
        cfg_kwargs["device"] = p.get("device", "cuda")

    if backend is not None:
        cfg_kwargs["backend"] = backend
    if resolution is not None:
        cfg_kwargs["resolution"] = resolution
    if frames is not None:
        cfg_kwargs["num_frames"] = frames
    if density is not None:
        cfg_kwargs["trajectory_density"] = density
    if device is not None:
        cfg_kwargs["device"] = device
    if output_dir is not None:
        cfg_kwargs["project_root"] = output_dir.parent
        cfg_kwargs["output_dir_name"] = output_dir.name

    cfg = MotionMirrorConfig(**cfg_kwargs)

    console.print(f"[bold]Motion Mirror[/bold] — backend=[cyan]{cfg.backend}[/cyan] "
                  f"res=[cyan]{cfg.resolution}[/cyan] frames=[cyan]{cfg.num_frames}[/cyan]")
    console.print(f"  image : {image}")
    console.print(f"  motion: {motion}")

    try:
        result = MotionMirrorPipeline(cfg).run(image, motion)
        console.print(f"\n[green]Done.[/green] Output: {result.output_path}")
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", style="bold")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}", style="bold")
        raise typer.Exit(code=1)


# ── download ──────────────────────────────────────────────────────────────────


@app.command()
def download(
    model: str = typer.Option(
        "all",
        help="Model(s) to download: all | dwpose | wan-move | dwpose-pose | dwpose-det.",
    ),
    cache_dir: Optional[Path] = typer.Option(None, help="Override default cache directory."),
    skip_check: bool = typer.Option(False, help="Skip disk-space preflight check."),
) -> None:
    """Download model weights to the local cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = MotionMirrorConfig()
    if cache_dir:
        object.__setattr__(cfg, "cache_dir", cache_dir)

    keys = _MODEL_GROUPS.get(model)
    if keys is None:
        if model in _MODEL_SPECS:
            keys = [model]
        else:
            console.print(f"[red]Unknown model:[/red] {model!r}")
            console.print(f"  Valid: {list(_MODEL_GROUPS.keys()) + list(_MODEL_SPECS.keys())}")
            raise typer.Exit(code=1)

    # Disk-space preflight
    if not skip_check:
        total_needed = sum(_MODEL_SPECS[k]["expected_bytes"] for k in keys)
        # cache_dir may not exist yet — check the nearest existing ancestor
        check_path = cfg.cache_dir
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent
        free = shutil.disk_usage(check_path).free
        if free < total_needed:
            needed_gb = total_needed / 1024 ** 3
            free_gb = free / 1024 ** 3
            console.print(
                f"[red]Insufficient disk space.[/red] "
                f"Need ~{needed_gb:.1f} GB, have {free_gb:.1f} GB free in {cfg.cache_dir}."
            )
            raise typer.Exit(code=1)

    for key in keys:
        spec = _MODEL_SPECS[key]
        dest_dir = cfg.model_cache(spec["cache_subdir"])
        label = spec["label"]

        if spec["filename"] is not None:
            dest_file = dest_dir / spec["filename"]
            if dest_file.exists() and dest_file.stat().st_size > 0:
                console.print(f"[dim]{label}[/dim] — [green]already cached[/green] ({dest_file})")
                continue
            console.print(f"Downloading [cyan]{label}[/cyan] …")
            try:
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    local_dir=str(dest_dir),
                )
                console.print(f"  [green]✓[/green] Saved to {dest_dir}")
            except Exception as exc:
                console.print(f"  [red]Failed:[/red] {exc}")
                raise typer.Exit(code=1)
        else:
            # Full repo snapshot
            if any(dest_dir.iterdir()) if dest_dir.exists() else False:
                console.print(f"[dim]{label}[/dim] — [green]already cached[/green] ({dest_dir})")
                continue
            console.print(f"Downloading [cyan]{label}[/cyan] — this may take a while …")
            try:
                snapshot_download(
                    repo_id=spec["repo_id"],
                    local_dir=str(dest_dir),
                )
                console.print(f"  [green]✓[/green] Saved to {dest_dir}")
            except Exception as exc:
                console.print(f"  [red]Failed:[/red] {exc}")
                raise typer.Exit(code=1)

    console.print("[green]Download complete.[/green]")


# ── presets ───────────────────────────────────────────────────────────────────


@app.command()
def presets(
    list_: bool = typer.Option(True, "--list", help="List all available presets."),
) -> None:
    """List available generation presets."""
    toml_files = sorted(_PRESETS_DIR.glob("*.toml"))
    if not toml_files:
        console.print("[yellow]No presets found.[/yellow]")
        return

    table = Table(title="Motion Mirror Presets", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Backend")
    table.add_column("Resolution")
    table.add_column("Frames", justify="right")
    table.add_column("Density", justify="right")
    table.add_column("Description")

    for f in toml_files:
        try:
            p = tomllib.loads(f.read_text(encoding="utf-8"))["preset"]
            table.add_row(
                p.get("name", f.stem),
                p.get("backend", "—"),
                p.get("resolution", "—"),
                str(p.get("num_frames", "—")),
                str(p.get("trajectory_density", "—")),
                p.get("description", ""),
            )
        except Exception:
            table.add_row(f.stem, "—", "—", "—", "—", "[red]parse error[/red]")

    console.print(table)


# ── benchmark ─────────────────────────────────────────────────────────────────


@app.command()
def benchmark(
    gpu_info: bool = typer.Option(False, "--gpu-info", help="Print GPU name and VRAM stats."),
) -> None:
    """Print system and GPU diagnostics."""
    import platform

    console.print(f"[bold]Motion Mirror[/bold] — system info")
    console.print(f"  Python  : {sys.version.split()[0]}")
    console.print(f"  Platform: {platform.system()} {platform.release()}")

    if gpu_info:
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                total_gb = props.total_memory / 1024 ** 3
                free_bytes, _ = torch.cuda.mem_get_info(idx)
                free_gb = free_bytes / 1024 ** 3
                used_gb = total_gb - free_gb

                console.print(f"\n  GPU     : {name}")
                console.print(f"  VRAM    : {total_gb:.1f} GB total, "
                               f"{used_gb:.1f} GB used, {free_gb:.1f} GB free")
                if total_gb < 22:
                    console.print(
                        f"\n  [yellow]Warning:[/yellow] Wan-Move-14B requires ~24 GB VRAM. "
                        f"Your GPU has {total_gb:.1f} GB. Use --backend controlnet for lighter inference."
                    )
                else:
                    console.print(f"\n  [green]✓[/green] VRAM sufficient for Wan-Move-14B.")
            else:
                console.print("\n  [yellow]No CUDA GPU detected.[/yellow]")
                console.print("  Wan-Move-14B requires a 24 GB+ NVIDIA GPU.")
        except ImportError:
            console.print("\n  [yellow]torch not installed.[/yellow]")
            console.print("  Run: pip install -r requirements-cuda.txt")
    else:
        console.print("\n  Run with [cyan]--gpu-info[/cyan] to check VRAM.")


# ── ui ────────────────────────────────────────────────────────────────────────


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host to bind the Gradio server."),
    port: int = typer.Option(7860, help="Port for the Gradio server."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Launch the Gradio web UI."""
    from .ui.app import create_app

    console.print(f"[bold]Motion Mirror UI[/bold] — http://{host}:{port}")
    demo = create_app()
    demo.launch(server_name=host, server_port=port, share=share)
