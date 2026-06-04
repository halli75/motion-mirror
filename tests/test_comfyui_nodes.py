from __future__ import annotations

import sys
import types
from pathlib import Path


def test_comfyui_node_mappings_include_v02b_nodes():
    import comfyui_nodes

    assert "MotionMirrorPoseExtract" in comfyui_nodes.NODE_CLASS_MAPPINGS
    assert "MotionMirrorTrajectoryGen" in comfyui_nodes.NODE_CLASS_MAPPINGS
    assert "MotionMirrorGenerate" in comfyui_nodes.NODE_CLASS_MAPPINGS


def test_motion_mirror_generate_routes_through_model_management(tmp_path, monkeypatch):
    from comfyui_nodes.nodes import MotionMirrorGenerate

    image = tmp_path / "char.png"
    motion = tmp_path / "motion.mp4"
    image.write_bytes(b"image")
    motion.write_bytes(b"video")
    calls: dict[str, object] = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return (str(tmp_path / "result.mp4"),)

    monkeypatch.setattr(
        "comfyui_nodes.nodes.model_management.run_motion_mirror_generation",
        fake_generate,
    )

    result = MotionMirrorGenerate().run(
        str(image),
        str(motion),
        "wan-1.3b-concat-id",
        "832x480",
        81,
        512,
        "cuda",
    )

    assert result == (str(tmp_path / "result.mp4"),)
    assert calls["backend"] == "wan-1.3b-concat-id"
    assert calls["image_path"] == str(image)
    assert calls["motion_video_path"] == str(motion)


def test_model_management_uses_comfy_interrupt_hook(monkeypatch):
    import comfyui_nodes.model_management as model_management

    interrupted = {"count": 0}

    def fake_interrupt_hook():
        interrupted["count"] += 1

    comfy_mod = types.ModuleType("comfy")
    comfy_mm_mod = types.ModuleType("comfy.model_management")
    comfy_mm_mod.throw_exception_if_processing_interrupted = fake_interrupt_hook
    comfy_mod.model_management = comfy_mm_mod
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_mm_mod)

    model_management.throw_if_interrupted()

    assert interrupted["count"] == 1
