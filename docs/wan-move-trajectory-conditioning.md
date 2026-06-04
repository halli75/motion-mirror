# Wan-Move Trajectory Conditioning Gap

Motion Mirror currently synthesizes dense trajectory maps, but the Wan I2V
generation paths do not yet inject those trajectories into the diffusion model.
They use trajectory metadata in prompt text as a temporary placeholder.

## Current Behavior

- `wan-move-14b`: uses Diffusers `WanImageToVideoPipeline`.
- `wan-move-gguf`: injects a GGUF transformer into Diffusers `WanImageToVideoPipeline`.
- `wan-move-fast`: uses LightX2V `task="i2v"`.

These paths do not pass `tracks` or `track_visibility` tensors to a Wan-Move
runtime.

## Upstream Wan-Move API Shape

The official Wan-Move repository uses a custom `wan.WanMove` runtime:

```python
wan_move = wan.WanMove(
    config=cfg,
    checkpoint_dir=ckpt_dir,
    device_id=device,
    rank=rank,
    t5_cpu=t5_cpu,
)

video = wan_move.generate(
    prompt,
    image,
    track,
    track_visibility,
    max_area=max_area,
    frame_num=frame_num,
    shift=sample_shift,
    sample_solver=sample_solver,
    sampling_steps=sample_steps,
    guide_scale=guide_scale,
    seed=seed,
    offload_model=offload_model,
)
```

## Required Adapter Work

1. Add an official Wan-Move backend adapter that lazy-imports `wan.WanMove`.
2. Download or resolve `Ruihang/Wan-Move-14B-480P` weights separately from
   vanilla Wan2.1 Diffusers weights.
3. Convert `TrajectoryMap.tracks` from normalized character-space coordinates to
   the pixel coordinate format expected by Wan-Move.
4. Add or derive `track_visibility` for every trajectory and frame.
5. Add non-GPU contract tests that assert `track` and `track_visibility` reach
   the backend call.
6. Add one real GPU smoke where changing the reference motion changes the track
   tensor while keeping character and prompt fixed.

Until those steps are done, documentation should describe the current Wan
Diffusers/GGUF/LightX2V paths as accessibility experiments rather than full
Wan-Move trajectory-guidance parity.
