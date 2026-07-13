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

V02A_BACKENDS = ("wan-1.3b-vace", "wan-14b-vace", "wan-14b-vace-gguf")


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
            resolution=args.resolution,
            frames=args.frames,
            steps=args.steps,
            density=args.density,
            seed=args.seed,
            offload_model=args.offload_model,
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
            "steps": args.steps,
            "density": args.density,
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
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps (1-200).")
    parser.add_argument("--density", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offload-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU-offload the model during generation (default: on, matches the historical smoke matrix).",
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
    resolution: str,
    frames: int,
    steps: int,
    density: int,
    seed: int,
    offload_model: bool = True,
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
            resolution=resolution,
            frames=frames,
            steps=steps,
            density=density,
            device="cuda" if torch.cuda.is_available() else "cpu",
            offload_model=offload_model,
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
    resolution: str,
    frames: int,
    steps: int,
    density: int,
    device: str,
    offload_model: bool = True,
) -> MotionMirrorConfig:
    """Build a config using only supported dataclass fields."""
    return MotionMirrorConfig(
        project_root=output_dir.parent,
        output_dir_name=output_dir.name,
        backend=backend,
        resolution=resolution,
        num_frames=frames,
        num_inference_steps=steps,
        trajectory_density=density,
        offload_model=offload_model,
        t5_cpu=True,
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


# Structural thresholds, tuned against synthetic fixtures in
# tests/test_v02a_gpu_smoke.py. Global luminance/variance stats proved
# orthogonal to content on real GPU evidence (passed skeleton-on-blue,
# rejected dancer-on-black), so the gate keys on foreground structure instead.
_MIN_FOREGROUND_AREA_FRACTION = 0.02
_MIN_FOREGROUND_FILL_RATIO = 0.35
_MAX_CONTROL_EDGE_IOU = 0.5


def _foreground_mask(frame) -> "Any":
    """Otsu-threshold the frame; foreground is the minority side of the split."""
    import numpy as np

    gray = cv2.cvtColor(np.asarray(frame, dtype="uint8"), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size // 2:
        mask = cv2.bitwise_not(mask)
    return mask


def _close_kernel_size(height: int, width: int) -> int:
    """Odd close-kernel size scaled to frame size (~0.3x the short side).

    The kernel must be generous enough to bridge the gaps *between* a stick
    figure's limbs so the closing recovers a solid silhouette — only then does a
    thin skeleton read as low fill. Scaling with resolution keeps the same
    discrimination on 160px fixtures and 832x480 GPU output alike.
    """
    size = max(15, int(min(height, width) * 0.3))
    return size + 1 if size % 2 == 0 else size


def _frame_foreground_stats(frame) -> tuple[float, float]:
    """Return (area fraction, fill ratio) of the largest foreground blob.

    Fill ratio = blob area / area of its morphological closing: dense rendered
    subjects stay near 1.0, thin-stroke skeletons collapse well below it.
    """
    import numpy as np

    mask = _foreground_mask(frame)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return 0.0, 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = np.where(labels == largest, 255, 0).astype("uint8")
    area = int(stats[largest, cv2.CC_STAT_AREA])
    k = _close_kernel_size(*mask.shape[:2])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed_area = int(np.count_nonzero(cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)))
    fill_ratio = area / closed_area if closed_area else 0.0
    return area / mask.size, fill_ratio


def _check_foreground(frames) -> tuple[bool, str]:
    """Require a real subject: a foreground blob that is both present and dense."""
    import numpy as np

    stats = [_frame_foreground_stats(frame) for frame in frames]
    area = float(np.median([s[0] for s in stats]))
    fill = float(np.median([s[1] for s in stats]))
    if area < _MIN_FOREGROUND_AREA_FRACTION:
        return False, f"no foreground subject (median area fraction {area:.3f})"
    if fill < _MIN_FOREGROUND_FILL_RATIO:
        return False, (
            f"foreground is thin strokes (fill ratio {fill:.2f}) "
            "— skeleton-like, not a rendered subject"
        )
    return True, f"foreground area {area:.2f}, fill ratio {fill:.2f}"


def _edge_iou(frame, control_frame) -> float:
    """IoU of dilated Canny edges; offset-invariant, resizes control to output."""
    import numpy as np

    gray = cv2.cvtColor(np.asarray(frame, dtype="uint8"), cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(
        np.asarray(control_frame, dtype="uint8"), cv2.COLOR_BGR2GRAY
    )
    if control_gray.shape != gray.shape:
        control_gray = cv2.resize(control_gray, (gray.shape[1], gray.shape[0]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), kernel) > 0
    control_edges = cv2.dilate(cv2.Canny(control_gray, 50, 150), kernel) > 0
    union = int(np.count_nonzero(edges | control_edges))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(edges & control_edges)) / union


def _check_control_passthrough(frames, control_frames) -> tuple[bool, str]:
    """Reject output whose edge structure matches the conditioning skeleton."""
    import numpy as np

    n = min(len(frames), len(control_frames))
    if n == 0:
        return True, "no control frames to compare"
    iou = float(np.median([_edge_iou(frames[i], control_frames[i]) for i in range(n)]))
    if iou > _MAX_CONTROL_EDGE_IOU:
        return False, (
            f"output edges match conditioning skeleton (edge IoU {iou:.2f}) "
            "— control passthrough, not a rendered character"
        )
    return True, f"edge IoU vs control {iou:.2f}"


def _check_content(
    output_path: Path, control_path: Path | None
) -> tuple[bool, str]:
    """Reject degenerate output: frozen, subject-less, or skeleton passthrough.

    A readable MP4 is not evidence of a rendered character. These thresholds are
    deliberately loose — they catch the gross failure modes (blank, frozen, or the
    control skeleton copied to the output), not subtle quality issues.
    """
    import numpy as np

    frames = _read_frames(output_path)
    if len(frames) < 2:
        return False, f"only {len(frames)} frame(s) decoded"
    stack = np.stack(frames)

    # 1. Actually moves: consecutive frames must differ.
    inter = float(np.abs(np.diff(stack, axis=0)).mean())
    if inter < 0.5:
        return False, f"static output (mean inter-frame diff {inter:.2f})"

    # 2. Has a dense foreground subject, not blank frames or thin strokes.
    foreground_ok, foreground_reason = _check_foreground(frames)
    if not foreground_ok:
        return False, foreground_reason

    # 3. Not the conditioning skeleton copied through.
    if control_path is not None:
        control_ok, control_reason = _check_control_passthrough(
            frames, _read_frames(control_path)
        )
        if not control_ok:
            return False, control_reason

    return True, f"content ok ({foreground_reason}, motion {inter:.1f})"


if __name__ == "__main__":
    raise SystemExit(main())
