# v0.2b Scope: Identity And Ecosystem

v0.2b should start only after the v0.2a hardware validation gate is either
passed or the README/hardware policy has been corrected to match measured
results.

## Goals

- Add identity preservation for the 1.3B path.
- Add a ComfyUI node pack that respects ComfyUI model management.
- Keep 14B identity preservation in v0.3 research, not v0.2b.

## Concat-ID Spike

Concat-ID targets Wan2.1 1.3B identity conditioning. The first task is a
compatibility spike:

1. Verify whether Concat-ID can augment the current `WanVACEPipeline` path.
2. If it cannot, ship identity as a separate experimental backend instead of
   mixing it into VACE.
3. Keep backend naming explicit, for example `wan-1.3b-concat-id`.

Likely files:

- `src/motion_mirror/config.py`
- `src/motion_mirror/generate/concat_id.py`
- `src/motion_mirror/generate/models.py`
- `src/motion_mirror/pipeline.py`
- `src/motion_mirror/cli.py`
- `src/motion_mirror/ui/app.py`
- `src/motion_mirror/presets/identity.toml`
- `tests/test_concat_id.py`

Non-GPU tests should cover missing dependency errors, missing weights, routing,
and fake runtime calls. GPU validation should compare `wan-1.3b-vace` and the
identity backend on the same inputs.

## ComfyUI Nodes

Create a top-level `comfyui_nodes/` package:

- `comfyui_nodes/__init__.py`
- `comfyui_nodes/nodes.py`
- `comfyui_nodes/model_management.py`
- `comfyui_nodes/README.md`

Initial nodes:

- `MotionMirrorPoseExtract`
- `MotionMirrorTrajectoryGen`
- `MotionMirrorGenerate`

Do not add `MotionMirrorFaceRestore` until CodeFormer exists in v0.3.

`MotionMirrorGenerate` must use ComfyUI model-management hooks instead of
direct CUDA model loading, otherwise it can bypass ComfyUI VRAM arbitration and
OOM when combined with other Wan nodes.

## Out Of Scope

- 14B identity preservation
- IPRO
- CodeFormer
- RIFE
- Concat-ID training workflows
- multi-identity generation
- automated MoveBench quality gates
