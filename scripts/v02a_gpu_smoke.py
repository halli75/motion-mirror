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
    content_ok: bool | None = None
    content_reason: str | None = None
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
        # A readable MP4 is necessary but not sufficient — the vace backend once
        # emitted the pose skeleton and still "passed". Verify the output has
        # real image content, actually moves, and is not the conditioning
        # skeleton copied through.
        control_path = output_dir / "conditioning_pose.mp4"
        content_ok, content_reason = _check_content(
            output_path, control_path if control_path.exists() else None
        )
        peak = (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        )
        return SmokeResult(
            backend=backend,
            ok=output_path.exists() and readable and content_ok,
            elapsed_s=round(time.perf_counter() - start, 2),
            output_path=str(output_path),
            readable_mp4=readable,
            frame_count=frame_count,
            peak_cuda_memory_gb=peak,
            content_ok=content_ok,
            content_reason=content_reason,
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


def _read_frames(path: Path):
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return [np.asarray(f, dtype="float32") for f in frames]


def _check_content(
    output_path: Path, control_path: Path | None
) -> tuple[bool, str]:
    """Reject degenerate output: solid/near-black, static, or skeleton passthrough.

    A readable MP4 is not evidence of a rendered character. These thresholds are
    deliberately loose — they catch the gross failure modes (blank, frozen, or the
    control skeleton copied to the output), not subtle quality issues.
    """
    import numpy as np

    frames = _read_frames(output_path)
    if len(frames) < 2:
        return False, f"only {len(frames)} frame(s) decoded"
    stack = np.stack(frames)

    # 1. Real image content: a rendered scene has broad spatial variance and is
    #    not almost entirely black. The skeleton-on-black failure has mean ~8/255.
    mean_lum = float(stack.mean())
    spatial_std = float(stack.reshape(len(frames), -1).std(axis=1).mean())
    if mean_lum < 20.0:
        return False, f"near-black output (mean luminance {mean_lum:.1f}/255)"
    if spatial_std < 12.0:
        return False, f"low spatial variance (std {spatial_std:.1f}) — likely blank/degenerate"

    # 2. Actually moves: consecutive frames must differ.
    inter = float(np.abs(np.diff(stack, axis=0)).mean())
    if inter < 0.5:
        return False, f"static output (mean inter-frame diff {inter:.2f})"

    # 3. Not the conditioning skeleton copied through.
    if control_path is not None:
        control = _read_frames(control_path)
        n = min(len(control), len(frames))
        if n:
            cstack = np.stack(control[:n])
            diff = float(np.abs(stack[:n] - cstack).mean())
            if diff < 6.0:
                return False, (
                    f"output matches conditioning skeleton (mean abs diff {diff:.1f}) "
                    "— control passthrough, not a rendered character"
                )
    return True, (
        f"content ok (mean {mean_lum:.0f}, spatial std {spatial_std:.0f}, "
        f"motion {inter:.1f})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
