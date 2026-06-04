from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_smoke_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "v02a_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("v02a_gpu_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_config_uses_supported_motion_mirror_config_fields(tmp_path):
    smoke = _load_smoke_module()

    cfg = smoke._build_config(
        backend="wan-1.3b-vace",
        output_dir=tmp_path / "smoke" / "vace",
        cache_dir=tmp_path / "cache",
        reference_masker="pose",
        resolution="832x480",
        frames=17,
        density=256,
        device="cpu",
    )

    assert cfg.project_root == tmp_path / "smoke"
    assert cfg.output_dir_name == "vace"
    assert cfg.output_dir == tmp_path / "smoke" / "vace"
    assert cfg.cache_dir == tmp_path / "cache"
    assert cfg.backend == "wan-1.3b-vace"
    assert cfg.num_frames == 17
    assert cfg.trajectory_density == 256
