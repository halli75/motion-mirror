# Motion Mirror ComfyUI Nodes

This directory is the Motion Mirror ComfyUI custom-node package.

## Nodes

- `MotionMirrorPoseExtract` - runs DWPose extraction on the motion video and
  outputs a pose artifact path (`.npz`). CPU-capable.
- `MotionMirrorTrajectoryGen` - loads the pose artifact, segments the character
  image, synthesizes the dense trajectory map, and outputs a trajectory
  artifact path (`.npz`). CPU-capable.
- `MotionMirrorGenerate` - runs generation. Accepts optional `pose_path` and
  `trajectory_path` inputs from the nodes above so extraction is not repeated;
  when omitted it runs the full pipeline end-to-end.

`MotionMirrorGenerate` routes through `comfyui_nodes/model_management.py` so
future Wan VACE model loading can cooperate with ComfyUI's model
management hooks instead of allocating CUDA memory directly. Intermediate
artifacts are written under ComfyUI's output directory (`motion_mirror/`).

## Install

Clone or copy this repository into a ComfyUI custom-nodes location, then install
Motion Mirror in the same Python environment:

```bash
pip install -e .
```

Install optional extras for the backend you plan to use, for example:

```bash
pip install -e ".[gpu-inference]"
```

## Scope

Real GPU workflow validation is deferred. Do not add a
`MotionMirrorFaceRestore` node; face restoration belongs to a later release.
