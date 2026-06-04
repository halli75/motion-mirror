# v0.2b Identity Backend Spike

## Decision

Use `wan-1.3b-concat-id` as a separate experimental backend. Do not fold
Concat-ID into `wan-1.3b-vace` in v0.2b.

## Rationale

The public Concat-ID Wan release targets `Wan2.1-T2V-1.3B` and a
DiffSynth-style `WanVideoPipeline` runtime. It loads Concat-ID identity weights
such as `first_stage.pt` or `second_stage_adaln.pt` and passes a reference face
image into the pipeline.

Motion Mirror's `wan-1.3b-vace` path uses `WanVACEPipeline` and skeleton/mask
conditioning. There is not enough public evidence that the released Concat-ID
Wan adapter can be safely loaded into VACE without a custom integration.

## Current Backend Contract

- Backend name: `wan-1.3b-concat-id`
- Preset: `identity`
- Download group: `concat-id` / `identity`
- Base weights: `Wan-AI/Wan2.1-T2V-1.3B`
- Adapter weights: `yongzhong/Concat-ID-Wan`
- Runtime: lazy DiffSynth imports through `src/motion_mirror/generate/concat_id.py`

The backend remains GPU-validation pending. Non-GPU CI covers config, routing,
asset checks, missing-dependency errors, and a fake runtime call.
