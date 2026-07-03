"""Measure wan-1.3b-vace peak VRAM under the 12 GB auto-tier config.

``hardware.py``'s 12 GB tier selects wan-1.3b-vace with ``t5_cpu=True`` but
``offload_model=False``, while ``scripts/v02a_gpu_smoke.py`` hardcodes both
flags on — so the smoke matrix never measures the config the 12 GB tier
actually ships. This probe fills that gap.

Runs ON the GPU pod after the main smoke matrix.

Usage:
    python vace_tier_probe.py --image X --motion Y --cache-dir Z \
        --output-dir OUT --report OUT/vace-tier-probe.json
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from motion_mirror import MotionMirrorConfig, MotionMirrorPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--density", type=int, default=256)
    args = parser.parse_args()

    import torch

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "probe": "wan-1.3b-vace 12GB-tier config (t5_cpu=True, offload_model=False)",
        "ok": False,
    }
    start = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        cfg = MotionMirrorConfig(
            project_root=args.output_dir.parent,
            output_dir_name=args.output_dir.name,
            backend="wan-1.3b-vace",
            resolution="832x480",
            num_frames=args.frames,
            trajectory_density=args.density,
            offload_model=False,
            t5_cpu=True,
            device="cuda",
            cache_dir=args.cache_dir,
        )
        result = MotionMirrorPipeline(cfg).run(args.image, args.motion)
        report.update(
            ok=result.output_path.exists(),
            output_path=str(result.output_path),
            peak_cuda_memory_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        )
    except Exception as exc:  # noqa: BLE001 - evidence path
        report.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            peak_cuda_memory_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        )
    report["elapsed_s"] = round(time.perf_counter() - start, 2)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
