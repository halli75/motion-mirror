# Motion Mirror Code Review (v0.2a)

Reviewed against the current repo plus `C:\Users\arnav\Downloads\motion_mirror_plan_v1.3.1.pdf`.

Non-GPU test status: `pytest -m "not gpu" -q` passes (`161 passed, 9 deselected`). That does not mean the advertised v0.2a behavior works.

## Findings

### Blocking

1. `wan-1.3b-vace` is exposed as a real backend, auto-selected by hardware detection, and routed by the pipeline, but the implementation is still a stub that raises `NotImplementedError`. This is not an edge case. It is a public code path. References: `src/motion_mirror/pipeline.py:83-90`, `src/motion_mirror/pipeline.py:129-132`, `src/motion_mirror/hardware.py:90-92`, `src/motion_mirror/generate/controlnet.py:47-51`.

2. `wan-move-fast` is also fake. The pipeline treats it as distinct, but the generator always loads the same 14B Wan checkpoint from `wan-move` and the same hard-coded HF model ID. There is no LightX2V-specific loading path. The backend name changes; the implementation does not. References: `src/motion_mirror/pipeline.py:129-130`, `src/motion_mirror/generate/wan_move.py:117`, `src/motion_mirror/generate/wan_move.py:148`, `src/motion_mirror/generate/wan_move.py:195`.

3. The trajectory stage mixes coordinate spaces. It feeds the character-image mask into camera stabilization and non-rigid flow sampling for the reference video. If the reference video and character image differ in size, crop, or framing, the background mask and flow seed region are wrong by construction. This undermines the main algorithmic claim in the spec. References: `src/motion_mirror/extract/trajectory.py:120`, `src/motion_mirror/extract/trajectory.py:136-140`, `src/motion_mirror/extract/trajectory.py:195-240`, `src/motion_mirror/extract/trajectory.py:513-555`.

### Major

4. `_build_body_transform()` does not use actual character bounds. It invents a centered box covering 80% of the image and maps the reference body into that fiction. The spec says to normalize against the real character body bounds. This implementation will misalign motion whenever the character is not centered or not full-frame. References: `src/motion_mirror/extract/trajectory.py:243-292`.

5. The code exposes VRAM tuning flags that mostly do nothing. `offload_model` and `t5_cpu` are accepted in config, CLI, presets, and auto-selection, but the 14B generator always enables sequential offload unconditionally and never reads either flag. That is dead configuration, not functionality. References: `src/motion_mirror/config.py:37-42`, `src/motion_mirror/cli.py:123-174`, `src/motion_mirror/hardware.py:71-92`, `src/motion_mirror/generate/wan_move.py:209-214`.

6. The trajectory compositor can randomly drop Layer-1 skeleton anchors. The code concatenates all layers and then randomly subsamples to the requested density. The spec describes skeleton tracks as the scaffold. A scaffold you randomly discard is not a scaffold. References: `src/motion_mirror/extract/trajectory.py:147-153`.

7. Character-image validation is promised but not implemented. `MultipleCharactersError` is exported, documented, and imported into segmentation, but the segmentation path never runs pose detection on the character image and never raises it. The public API and README overstate what the code actually checks. References: `src/motion_mirror/extract/segment.py:21`, `src/motion_mirror/extract/segment.py:110-111`, `src/motion_mirror/__init__.py:2-14`, `README.md:192`.

8. `backend='auto'` falls back to `wan-move-14b` when no CUDA GPU is detected. That is a bad default. On a CPU-only machine it resolves to a backend the machine cannot actually use, instead of failing clearly or falling back to a safe mode. References: `src/motion_mirror/hardware.py:129-139`.

### Medium

9. Packaging is sloppy. Presets are loaded from a repo-relative `presets/` directory, but the setuptools config only packages Python modules from `src/`. Editable installs hide this. A real wheel install is likely to lose preset files. References: `src/motion_mirror/cli.py:34-44`, `src/motion_mirror/cli.py:290-305`, `pyproject.toml:55-58`.

10. The UI and tests are giving false confidence. The UI still exposes deprecated `controlnet` and hides the new backend names. The "controlnet integration" test explicitly runs `backend="mock"` and never touches the real route. The UI tests do not call the actual callback; they reimplement it inline. That is test-shaped theater. References: `src/motion_mirror/ui/app.py:81-85`, `tests/test_pipeline_integration.py:137-151`, `tests/test_ui.py:52-76`, `tests/test_ui.py:79-104`.

## Open Questions / Assumptions

- This review assumes v0.2a is intentionally incomplete, but exposed CLI/UI/docs paths are still fair game. Incomplete is acceptable. Misrepresented is not.
- I did not run GPU paths. The current review is based on code inspection and the passing non-GPU suite.

## Overall Verdict

The repo is ahead on marketing and behind on implementation. The core problem is not that v0.2a is unfinished. The core problem is that unfinished features are already wired into public surfaces as if they exist, while the tests mostly validate mocks, synthetic happy paths, and duplicated test logic.

Clean this up before adding more features. Right now the codebase is carrying more narrative than truth.
