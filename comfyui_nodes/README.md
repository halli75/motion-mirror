# Motion Mirror ComfyUI Nodes

This directory is the initial v0.2b ComfyUI custom-node scaffold.

## Nodes

- `MotionMirrorPoseExtract`
- `MotionMirrorTrajectoryGen`
- `MotionMirrorGenerate`

`MotionMirrorGenerate` routes through `comfyui_nodes/model_management.py` so
future Wan and Concat-ID model loading can cooperate with ComfyUI's model
management hooks instead of allocating CUDA memory directly.

## Install

Clone or copy this repository into a ComfyUI custom-nodes location, then install
Motion Mirror in the same Python environment:

```bash
pip install -e .
```

Install optional extras for the backend you plan to use, for example:

```bash
pip install -e ".[concat-id]"
```

## Scope

This is scaffolding only. Real GPU workflow validation is deferred. Do not add a
`MotionMirrorFaceRestore` node in v0.2b; CodeFormer belongs to v0.3 scope.
