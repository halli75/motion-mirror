"""Manual GPU smoke tests for release validation runs.

These tests are intentionally marked ``gpu`` and are not part of normal CI.
They require a CUDA machine, downloaded backend weights, and real input media.

Example:
    $env:MOTION_MIRROR_GPU_IMAGE="C:\\inputs\\character.png"
    $env:MOTION_MIRROR_GPU_MOTION="C:\\inputs\\motion.mp4"
    $env:MOTION_MIRROR_GPU_BACKENDS="wan-1.3b-vace"
    pytest -m gpu tests/test_gpu_smoke_runs.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_gpu_smoke_script():
    module_path = _repo_root() / "scripts" / "v02a_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("v02a_gpu_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for manual GPU smoke tests")


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for manual GPU smoke tests")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _assert_readable_video(path: Path) -> int:
    assert path.exists(), f"Output video does not exist: {path}"
    assert path.stat().st_size > 0, f"Output video is empty: {path}"
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened(), f"OpenCV could not open output video: {path}"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        assert ok, f"OpenCV could not read first frame from: {path}"
        assert frame_count > 0, f"Output video has no frames: {path}"
        return frame_count
    finally:
        cap.release()


@pytest.mark.gpu
def test_v02a_gpu_smoke_runner_matrix_writes_success_report(tmp_path):
    """Run the real v0.2a GPU validation script and validate its JSON evidence."""
    _require_cuda()
    image = _env_path("MOTION_MIRROR_GPU_IMAGE")
    motion = _env_path("MOTION_MIRROR_GPU_MOTION")
    backends = [
        item.strip()
        for item in os.environ.get("MOTION_MIRROR_GPU_BACKENDS", "wan-1.3b-vace").split(",")
        if item.strip()
    ]
    cache_dir = os.environ.get("MOTION_MIRROR_GPU_CACHE_DIR")
    report = tmp_path / "v02a-gpu-smoke" / "report.json"
    output_dir = tmp_path / "v02a-gpu-smoke" / "outputs"

    argv = [
        "--image",
        str(image),
        "--motion",
        str(motion),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report),
        "--resolution",
        os.environ.get("MOTION_MIRROR_GPU_RESOLUTION", "832x480"),
        "--frames",
        os.environ.get("MOTION_MIRROR_GPU_FRAMES", "17"),
        "--density",
        os.environ.get("MOTION_MIRROR_GPU_DENSITY", "256"),
    ]
    for backend in backends:
        argv.extend(["--backend", backend])
    if cache_dir:
        argv.extend(["--cache-dir", cache_dir])
    if os.environ.get("MOTION_MIRROR_GPU_REFERENCE_MASKER") == "sam2":
        argv.extend(["--reference-masker", "sam2"])

    smoke = _load_gpu_smoke_script()
    exit_code = smoke.main(argv)

    assert exit_code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["cuda"]["available"] is True
    assert [result["backend"] for result in data["results"]] == backends
    for result in data["results"]:
        assert result["ok"] is True
        assert result["readable_mp4"] is True
        assert result["peak_cuda_memory_gb"] is not None
        assert result["peak_cuda_memory_gb"] > 0
        _assert_readable_video(Path(result["output_path"]))
