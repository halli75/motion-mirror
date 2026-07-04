# ComfyUI Node Pack Scope

Motion Mirror ships a ComfyUI node pack so the VACE pipeline can be used inside
ComfyUI graphs while respecting ComfyUI's own model management.

> Note: the earlier Concat-ID 1.3B identity backend explored in this document has
> been dropped. Motion Mirror's backend lineup is VACE-only — `wan-1.3b-vace` is
> the model. Reference-image identity adherence is loose at 1.3B scale and is a
> known limitation, not a separate backend. See the README's Known Limitations.

## Goals

- Provide a ComfyUI node pack that respects ComfyUI model management.
- Expose the existing pose extraction, trajectory generation, and VACE
  generation stages as reusable nodes.

## ComfyUI Nodes

Top-level `comfyui_nodes/` package:

- `comfyui_nodes/__init__.py`
- `comfyui_nodes/nodes.py`
- `comfyui_nodes/model_management.py`
- `comfyui_nodes/README.md`

Initial nodes:

- `MotionMirrorPoseExtract`
- `MotionMirrorTrajectoryGen`
- `MotionMirrorGenerate`

Do not add `MotionMirrorFaceRestore` until face restoration exists in a later
release.

`MotionMirrorGenerate` must use ComfyUI model-management hooks instead of
direct CUDA model loading, otherwise it can bypass ComfyUI VRAM arbitration and
OOM when combined with other Wan nodes.

## Out Of Scope

- 14B identity preservation
- IPRO
- CodeFormer
- RIFE
- multi-identity generation
- automated MoveBench quality gates
</content>
