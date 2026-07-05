"""Motion Mirror command-line interface."""
from __future__ import annotations

import importlib.resources
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
    help="Motion Mirror - local-first motion transfer.",
    no_args_is_help=True,
)
console = Console()

_PRESETS_DIR = importlib.resources.files("motion_mirror") / "presets"


def _list_preset_files() -> list:
    """Return sorted list of preset Traversable entries (.toml files)."""
    return sorted(
        (p for p in _PRESETS_DIR.iterdir() if p.name.endswith(".toml") and p.is_file()),
        key=lambda p: p.name,
    )


def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.toml"
    if not path.is_file():
        available = [p.name.rsplit(".", 1)[0] for p in _list_preset_files()]
        raise typer.BadParameter(
            f"Preset {name!r} not found. Available: {available}"
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))["preset"]


def _is_spec_cached(dest_dir: Path, spec: dict) -> bool:
    if not dest_dir.exists():
        return False

    filename = spec.get("filename")
    if filename is not None and not _cache_path_complete(dest_dir / filename):
        return False

    min_cached_bytes = spec.get("min_cached_bytes")
    if min_cached_bytes is not None:
        return _cache_size_bytes(dest_dir) >= int(min_cached_bytes)

    if filename is not None:
        return True
    return any(dest_dir.iterdir())


def _cache_path_complete(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def _cache_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


# Safety margin applied to the disk-space preflight: require the free space to
# exceed the sum of expected download sizes by this factor so a download cannot
# fill the disk to the brim (temp files, metadata, .incomplete artifacts).
_DISK_MARGIN = 1.1

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
    "wan-1.3b-vace": {
        "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "filename": None,
        "expected_bytes": 19_000_000_000,
        "min_cached_bytes": 3_000_000_000,
        "cache_subdir": "wan-1.3b-vace",
        "label": "Wan2.1-VACE-1.3B (lightweight, ~19 GB download, needs ~9 GB VRAM) [backend: wan-1.3b-vace]",
    },
    "sam2": {
        "repo_id": "facebook/sam2-hiera-large",
        "filename": None,
        "expected_bytes": 1_800_000_000,
        "min_cached_bytes": 500_000_000,
        "cache_subdir": "sam2",
        "label": "SAM-2 Large image segmenter (~1.8 GB) [--segmenter sam2]",
    },
    "wan-14b-vace": {
        "repo_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "filename": None,
        "expected_bytes": 76_000_000_000,
        "min_cached_bytes": 60_000_000_000,
        "cache_subdir": "wan-14b-vace",
        "label": "Wan2.1-VACE-14B full (~75 GB, GPU-unvalidated, explicit --backend only) [backend: wan-14b-vace]",
    },
    "wan-14b-vace-gguf": {
        "repo_id": "QuantStack/Wan2.1_14B_VACE-GGUF",
        "filename": "Wan2.1_14B_VACE-Q4_K_M.gguf",
        "expected_bytes": 11_600_000_000,
        "cache_subdir": "wan-14b-vace-gguf",
        "label": "Wan2.1-VACE-14B Q4_K_M GGUF transformer (~11.6 GB) [backend: wan-14b-vace-gguf]",
    },
    "wan-14b-vace-base": {
        "repo_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "filename": None,
        "expected_bytes": 12_000_000_000,
        "min_cached_bytes": 9_000_000_000,
        "cache_subdir": "wan-14b-vace-base",
        "allow_patterns": ["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "model_index.json"],
        "label": "Wan2.1-VACE-14B base components sans transformer (~12 GB, for the GGUF backend)",
    },
}

_MODEL_GROUPS = {
    "dwpose": ["dwpose-pose", "dwpose-det"],
    "vace": ["wan-1.3b-vace"],
    "vace-14b": ["wan-14b-vace"],
    "vace-14b-gguf": ["wan-14b-vace-gguf", "wan-14b-vace-base"],
    "extras": ["sam2"],
    # "all" is deliberately the validated ~6 GB lineup only. The 14B backends
    # (~75 GB full, ~24 GB GGUF+base) are GPU-unvalidated and must be requested
    # explicitly via their own model/group keys - growing "all" to ~130 GB
    # silently would be hostile.
    "all": [
        "dwpose-pose",
        "dwpose-det",
        "wan-1.3b-vace",
        "sam2",
    ],
}


@app.command()
def run(
    image: Path = typer.Argument(..., help="Character image path (PNG/JPG/WEBP)."),
    motion: Path = typer.Argument(..., help="Reference motion video path (MP4/MOV/AVI/MKV)."),
    backend: Optional[str] = typer.Option(None, help="Backend: auto | wan-1.3b-vace | wan-14b-vace | wan-14b-vace-gguf | mock (14B backends GPU-unvalidated)."),
    resolution: Optional[str] = typer.Option(None, help="Output resolution WxH, e.g. 832x480."),
    frames: Optional[int] = typer.Option(None, help="Number of output frames."),
    density: Optional[int] = typer.Option(None, help="Trajectory density (512 = default, 1024 = HQ)."),
    device: Optional[str] = typer.Option(None, help="Compute device: cuda | cpu."),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: ./outputs)."),
    preset: Optional[str] = typer.Option(None, help="Load settings from a preset name."),
    offload_model: bool = typer.Option(False, "--offload-model", help="Layer-by-layer CPU offload (saves VRAM, slower)."),
    t5_cpu: bool = typer.Option(False, "--t5-cpu", help="Keep T5 text encoder on CPU (~12 GB VRAM saved)."),
    flow_estimator: Optional[str] = typer.Option(None, "--flow-estimator", help="Optical flow backend: farneback | raft."),
    segmenter: Optional[str] = typer.Option(None, "--segmenter", help="Segmentation model: rembg | sam2."),
    auto: bool = typer.Option(False, "--auto", help="Auto-select backend from available VRAM."),
) -> None:
    """Run the full motion transfer pipeline."""
    cfg_kwargs: dict = {}
    if preset:
        preset_data = _load_preset(preset)
        cfg_kwargs["backend"] = preset_data.get("backend", "wan-1.3b-vace")
        cfg_kwargs["resolution"] = preset_data.get("resolution", "832x480")
        cfg_kwargs["num_frames"] = preset_data.get("num_frames", 81)
        cfg_kwargs["trajectory_density"] = preset_data.get("trajectory_density", 512)
        cfg_kwargs["device"] = preset_data.get("device", "cuda")
        if "offload_model" in preset_data:
            cfg_kwargs["offload_model"] = preset_data["offload_model"]
        if "t5_cpu" in preset_data:
            cfg_kwargs["t5_cpu"] = preset_data["t5_cpu"]
        if "flow_estimator" in preset_data:
            cfg_kwargs["flow_estimator"] = preset_data["flow_estimator"]
        if "segmenter" in preset_data:
            cfg_kwargs["segmenter"] = preset_data["segmenter"]

    if auto:
        cfg_kwargs["backend"] = "auto"
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
    if offload_model:
        cfg_kwargs["offload_model"] = True
    if t5_cpu:
        cfg_kwargs["t5_cpu"] = True
    if flow_estimator is not None:
        cfg_kwargs["flow_estimator"] = flow_estimator
    if segmenter is not None:
        cfg_kwargs["segmenter"] = segmenter

    cfg = MotionMirrorConfig(**cfg_kwargs)

    console.print(
        f"[bold]Motion Mirror[/bold] - backend=[cyan]{cfg.backend}[/cyan] "
        f"res=[cyan]{cfg.resolution}[/cyan] frames=[cyan]{cfg.num_frames}[/cyan]"
    )
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


@app.command()
def download(
    model: str = typer.Option(
        "all",
        help=(
            "Model(s) to download: all | dwpose | vace | vace-14b | "
            "vace-14b-gguf | extras | wan-1.3b-vace | wan-14b-vace | "
            "wan-14b-vace-gguf | wan-14b-vace-base | sam2 | dwpose-pose | "
            "dwpose-det. NOTE: 'all' is the validated ~6 GB lineup only; the "
            "~75 GB / ~24 GB 14B backends must be requested explicitly."
        ),
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

    if not skip_check:
        total_needed = sum(_MODEL_SPECS[key]["expected_bytes"] for key in keys)
        required_with_margin = int(total_needed * _DISK_MARGIN)
        check_path = cfg.cache_dir
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent
        free = shutil.disk_usage(check_path).free
        if free < required_with_margin:
            needed_gb = required_with_margin / 1024 ** 3
            free_gb = free / 1024 ** 3
            console.print(
                f"[red]Insufficient disk space.[/red] "
                f"Need ~{needed_gb:.1f} GB (incl. {_DISK_MARGIN:g}x safety margin), "
                f"have {free_gb:.1f} GB free in {cfg.cache_dir}."
            )
            raise typer.Exit(code=1)

    for key in keys:
        spec = _MODEL_SPECS[key]
        dest_dir = cfg.model_cache(spec["cache_subdir"])
        label = spec["label"]
        already_cached = _is_spec_cached(dest_dir, spec)

        if spec["filename"] is not None:
            dest_file = dest_dir / spec["filename"]
            if dest_file.exists() and dest_file.stat().st_size > 0:
                console.print(f"[dim]{label}[/dim] - [green]already cached[/green] ({dest_file})")
                continue
            console.print(f"Downloading [cyan]{label}[/cyan] ...")
            try:
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    local_dir=str(dest_dir),
                )
                console.print(f"  [green]ok[/green] Saved to {dest_dir}")
            except Exception as exc:
                console.print(f"  [red]Failed:[/red] {exc}")
                raise typer.Exit(code=1)
        else:
            if already_cached:
                console.print(f"[dim]{label}[/dim] - [green]already cached[/green] ({dest_dir})")
                continue
            console.print(f"Downloading [cyan]{label}[/cyan] - this may take a while ...")
            try:
                snapshot_download(
                    repo_id=spec["repo_id"],
                    local_dir=str(dest_dir),
                    allow_patterns=spec.get("allow_patterns"),
                )
                console.print(f"  [green]ok[/green] Saved to {dest_dir}")
            except Exception as exc:
                console.print(f"  [red]Failed:[/red] {exc}")
                raise typer.Exit(code=1)

    console.print("[green]Download complete.[/green]")


@app.command()
def presets(
    list_: bool = typer.Option(True, "--list", help="List all available presets."),
) -> None:
    """List available generation presets."""
    toml_files = _list_preset_files()
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

    for path in toml_files:
        try:
            preset_data = tomllib.loads(path.read_text(encoding="utf-8"))["preset"]
            table.add_row(
                preset_data.get("name", path.stem),
                preset_data.get("backend", "-"),
                preset_data.get("resolution", "-"),
                str(preset_data.get("num_frames", "-")),
                str(preset_data.get("trajectory_density", "-")),
                preset_data.get("description", ""),
            )
        except Exception:
            table.add_row(path.stem, "-", "-", "-", "-", "[red]parse error[/red]")

    console.print(table)


@app.command()
def benchmark(
    gpu_info: bool = typer.Option(False, "--gpu-info", help="Print GPU name and VRAM stats."),
) -> None:
    """Print system and GPU diagnostics."""
    import platform

    console.print("[bold]Motion Mirror[/bold] - system info")
    console.print(f"  Python  : {sys.version.split()[0]}")
    console.print(f"  Platform: {platform.system()} {platform.release()}")

    if gpu_info:
        from .hardware import InsufficientVRAMError, get_gpu_info, recommend_backend

        info = get_gpu_info()
        if info is None:
            console.print("\n  [yellow]No CUDA GPU detected.[/yellow]")
            console.print("  Real generation requires a CUDA GPU with 8+ GB VRAM.")
        else:
            console.print(f"\n  GPU     : {info.name}")
            console.print(
                f"  VRAM    : {info.total_vram_gb:.1f} GB total, "
                f"{info.used_vram_gb:.1f} GB used, "
                f"{info.free_vram_gb:.1f} GB free"
            )
            try:
                backend_rec, overrides = recommend_backend(info.free_vram_gb)
                override_str = (
                    "  (" + ", ".join(f"--{k.replace('_', '-')}" for k in overrides) + ")"
                    if overrides else ""
                )
                console.print(
                    f"\n  [green]Recommended backend:[/green] "
                    f"[cyan]{backend_rec}[/cyan]{override_str}"
                )
                console.print(f"  Run: [dim]motion-mirror run --backend {backend_rec} ...[/dim]")
            except InsufficientVRAMError as exc:
                console.print(
                    f"\n  [red]Insufficient VRAM:[/red] {exc.available_gb:.1f} GB free, "
                    f"need {exc.required_gb:.0f} GB minimum."
                )
    else:
        console.print("\n  Run with [cyan]--gpu-info[/cyan] to check VRAM.")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host to bind the Gradio server."),
    port: int = typer.Option(7860, help="Port for the Gradio server."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Launch the Gradio web UI."""
    from .ui.app import create_app

    console.print(f"[bold]Motion Mirror UI[/bold] - http://{host}:{port}")
    demo = create_app()
    demo.launch(server_name=host, server_port=port, share=share)
