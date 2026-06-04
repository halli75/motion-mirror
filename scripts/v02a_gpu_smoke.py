"""Run v0.2a GPU smoke validation and write machine-readable evidence.

This script is intentionally outside the normal pytest suite because it needs
CUDA hardware, downloaded model weights, and several optional backend packages.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline

V02A_BACKENDS = (
    "wan-1.3b-vace",
    "wan-move-fast",
    "wan-move-gguf",
    "wan-move-14b",
)


@dataclass(slots=True)
class SmokeResult:
    backend: str
    ok: bool
    elapsed_s: float
    output_path: str | None
    readable_mp4: bool
    frame_count: int | None
    peak_cuda_memory_gb: float | None
    error: str | None = None
    traceback: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cuda = _cuda_info()
    if not cuda["available"] and not args.allow_cpu:
        report = {
            "ok": False,
            "reason": "CUDA is not available; pass --allow-cpu only for harness debugging.",
            "cuda": cuda,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    results = [
        _run_backend(
            backend=backend,
            image=args.image,
            motion=args.motion,
            output_dir=args.output_dir / backend,
            cache_dir=args.cache_dir,
            reference_masker=args.reference_masker,
            resolution=args.resolution,
            frames=args.frames,
            density=args.density,
            seed=args.seed,
        )
        for backend in args.backend
    ]
    report = {
        "ok": all(result.ok for result in results),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda": cuda,
        "inputs": {
            "image": str(args.image),
            "motion": str(args.motion),
            "resolution": args.resolution,
            "frames": args.frames,
            "density": args.density,
            "reference_masker": args.reference_masker,
        },
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Character image path.")
    parser.add_argument("--motion", type=Path, required=True, help="Reference motion video path.")
    parser.add_argument(
        "--backend",
        choices=V02A_BACKENDS,
        action="append",
        required=True,
        help="Backend to validate. Repeat to run a matrix.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v02a-smoke"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("outputs/v02a-smoke/report.json"))
    parser.add_argument("--resolution", default="832x480")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--density", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-masker",
        choices=("pose", "sam2"),
        default="pose",
        help="Use sam2 for the dedicated SAM-2 reference-mask propagation smoke.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without CUDA for harness debugging only.",
    )
    return parser.parse_args(argv)


def _cuda_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return {"torch_installed": False, "available": False, "devices": []}

    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                }
            )
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "available": torch.cuda.is_available(),
        "devices": devices,
    }


def _run_backend(
    *,
    backend: str,
    image: Path,
    motion: Path,
    output_dir: Path,
    cache_dir: Path | None,
    reference_masker: str,
    resolution: str,
    frames: int,
    density: int,
    seed: int,
) -> SmokeResult:
    start = time.perf_counter()
    output_path: Path | None = None
    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        cfg = _build_config(
            backend=backend,
            output_dir=output_dir,
            cache_dir=cache_dir,
            reference_masker=reference_masker,
            resolution=resolution,
            frames=frames,
            density=density,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = MotionMirrorPipeline(cfg).run(image, motion)
        output_path = result.output_path
        readable, frame_count = _probe_video(output_path)
        peak = (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        )
        return SmokeResult(
            backend=backend,
            ok=output_path.exists() and readable,
            elapsed_s=round(time.perf_counter() - start, 2),
            output_path=str(output_path),
            readable_mp4=readable,
            frame_count=frame_count,
            peak_cuda_memory_gb=peak,
        )
    except Exception as exc:  # pragma: no cover - evidence path for manual GPU runs
        return SmokeResult(
            backend=backend,
            ok=False,
            elapsed_s=round(time.perf_counter() - start, 2),
            output_path=str(output_path) if output_path else None,
            readable_mp4=False,
            frame_count=None,
            peak_cuda_memory_gb=None,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def _build_config(
    *,
    backend: str,
    output_dir: Path,
    cache_dir: Path | None,
    reference_masker: str,
    resolution: str,
    frames: int,
    density: int,
    device: str,
) -> MotionMirrorConfig:
    """Build a config using only supported dataclass fields."""
    return MotionMirrorConfig(
        project_root=output_dir.parent,
        output_dir_name=output_dir.name,
        backend=backend,
        resolution=resolution,
        num_frames=frames,
        trajectory_density=density,
        offload_model=True,
        t5_cpu=True,
        reference_masker=reference_masker,  # type: ignore[arg-type]
        device=device,
        cache_dir=cache_dir or MotionMirrorConfig().cache_dir,
    )


def _probe_video(path: Path) -> tuple[bool, int | None]:
    if not path.exists():
        return False, None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        return bool(ok), frame_count
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
