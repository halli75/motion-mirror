# Motion Mirror Deep Code Review (v0.2a)

This is the detailed review to complement `CODE_REVIEW_v0.2a.md`.

Scope reviewed:

`src/`, `tests/`, `presets/`, `pyproject.toml`, packaging metadata, and README/spec alignment against `C:\Users\arnav\Downloads\motion_mirror_plan_v1.3.1.pdf`.

Baseline:

`pytest -m "not gpu" -q` currently passes with `161 passed, 9 deselected`.

That result is not a quality signal on its own. Most non-GPU coverage is shape-level, mock-driven, or duplicated test logic. The review below is intentionally harsh because the codebase is overselling itself.

## Highest-Severity Findings

1. The real 14B generation path does not use trajectories as motion conditioning. It loads the `.npz`, reads `density`, turns that into prompt text, and then calls plain `WanImageToVideoPipeline`. The central product claim is motion transfer, but the implementation is image-to-video plus a prompt string. References: `src/motion_mirror/generate/wan_move.py:233-253`. Suggested change: either remove all motion-control claims until conditioning exists, or block the feature behind a hard "not implemented" error instead of pretending trajectory synthesis is part of the generation path.

2. `wan-1.3b-vace` is exposed in config, CLI, auto-detection, presets, and pipeline routing, but the implementation is still a stub that raises `NotImplementedError`. This is a public lie, not a private TODO. References: `src/motion_mirror/config.py:23-30`, `src/motion_mirror/cli.py:73-78`, `src/motion_mirror/hardware.py:89-92`, `src/motion_mirror/pipeline.py:129-132`, `src/motion_mirror/generate/controlnet.py:47-52`. Suggested change: remove the backend from all public surfaces until it exists, or implement the real path before exposing it.

3. `wan-move-fast` is fake. The CLI downloads a LightX2V asset, but the pipeline still routes into the normal 14B generator and the generator always loads the same `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` source from the `wan-move` cache. References: `src/motion_mirror/cli.py:80-85`, `src/motion_mirror/pipeline.py:129-130`, `src/motion_mirror/generate/wan_move.py:117`, `src/motion_mirror/generate/wan_move.py:148`, `src/motion_mirror/generate/wan_move.py:195`. Suggested change: either create a distinct fast backend loader and artifact layout, or remove `wan-move-fast` from config, presets, and docs.

4. The trajectory code mixes character-image space and reference-video space. The character segmentation mask is used for camera-motion compensation and non-rigid seed selection on the video frames. If the image and video differ in crop, aspect ratio, or framing, the mask is wrong by construction. References: `src/motion_mirror/extract/trajectory.py:120`, `src/motion_mirror/extract/trajectory.py:136-140`, `src/motion_mirror/extract/trajectory.py:195-240`, `src/motion_mirror/extract/trajectory.py:513-555`. Suggested change: split the data model into character mask, reference-frame subject ROI, and transform objects. Do not reuse character-image masks as video-space masks.

5. The tests do not defend the product claims. They validate mocks, happy-path shapes, and duplicated helper logic while the core behaviors above remain broken or unimplemented. References: `tests/test_pipeline_integration.py:1-6`, `tests/test_ui.py:52-112`, `tests/test_generate.py:124-178`, `tests/test_trajectory.py:102-312`. Suggested change: stop calling the mock suite the primary CI quality gate. Add real behavior assertions or tone down claims.

## File-by-File Deep Dive

## `pyproject.toml`

1. `version = "0.1.0"` while the branch is clearly carrying v0.2a surfaces and docs. If this is intentional, fine. If not, versioning is stale. Reference: `pyproject.toml:7`. Suggested change: either keep the package version aligned with the public feature surface or explicitly document that v0.2a is an unreleased branch state.

2. Presets are not packaged. The setuptools config only packages Python code from `src`, and the generated `src/motion_mirror.egg-info/SOURCES.txt` does not include `presets/*.toml`. The CLI loads presets from a repo-relative directory, so editable installs work while wheel installs likely break. References: `pyproject.toml:54-58`, `src/motion_mirror.egg-info/SOURCES.txt`, `src/motion_mirror/cli.py:34-44`. Suggested change: package presets as package data and load them through `importlib.resources`.

3. Optional dependency groups are drifting away from runtime truth. `gpu-inference` exists, but `wan-move-fast` and `sam2` still require manual GitHub installs and are not actually wired into runtime cleanly. References: `pyproject.toml:39-49`, `src/motion_mirror/extract/segment.py:64-69`, `src/motion_mirror/generate/wan_move.py:181-208`. Suggested change: define one coherent dependency story per backend, or stop advertising partial install workflows.

## `src/motion_mirror/__init__.py`

1. This file is mostly fine. It is a simple re-export surface.

2. The problem is not the file itself. The problem is that it exports exceptions and hardware helpers for behaviors that are not consistently implemented elsewhere, especially `MultipleCharactersError` and auto backend selection. References: `src/motion_mirror/__init__.py:1-18`, `src/motion_mirror/extract/segment.py:21`, `src/motion_mirror/hardware.py:128-139`. Suggested change: keep `__init__.py` thin, but audit exported symbols against actual runtime behavior.

## `src/motion_mirror/config.py`

1. The config is stringly typed and under-validated. `backend`, `resolution`, `device`, `flow_estimator`, and `segmenter` are all plain strings with no runtime validation beyond whatever happens later. `Literal` helps static tooling, not runtime. References: `src/motion_mirror/config.py:23-42`, `src/motion_mirror/config.py:53-56`. Suggested change: add `__post_init__` validation or replace strings with enums/value objects.

2. `resolution_wh` throws raw `ValueError` from string splitting. That is lazy error handling disguised as an API. References: `src/motion_mirror/config.py:53-56`. Suggested change: validate `resolution` on construction and raise a domain-specific error message.

3. `model_cache()` creates directories as a side effect of a getter-like method. That is convenient but sloppy because read paths and mutation are mixed. Reference: `src/motion_mirror/config.py:58-61`. Suggested change: keep cache path computation pure and move directory creation into call sites that actually download or write.

4. `offload_model` and `t5_cpu` exist in config but are not honored by the real 14B generator. The config surface is ahead of the implementation. References: `src/motion_mirror/config.py:36-42`, `src/motion_mirror/generate/wan_move.py:209-214`. Suggested change: either wire the flags through or remove them until they have behavior.

## `src/motion_mirror/exceptions.py`

1. The hierarchy itself is clean enough.

2. The docstrings overpromise unsupported behavior. `MultiplePeopleDetectedError` documents `--person-index N`, but no such CLI option exists. References: `src/motion_mirror/exceptions.py:81-95`, `src/motion_mirror/extract/pose.py:206-210`, `src/motion_mirror/cli.py:111-129`. Suggested change: either implement `--person-index` or remove it from error and doc text.

3. `VideoDecodeError` can carry `ffmpeg_output`, but the actual decode path in `extract_pose()` relies on OpenCV and never fills that field. The exception API is richer than its usage. References: `src/motion_mirror/exceptions.py:46-58`, `src/motion_mirror/extract/pose.py:63-91`. Suggested change: either populate the field through an ffmpeg-backed decode path or simplify the exception.

## `src/motion_mirror/types.py`

1. The dataclasses carry no invariants. Shapes, dtypes, frame counts, and value ranges are all "trust me." That is fine for prototypes and bad for public interfaces. References: `src/motion_mirror/types.py:9-81`. Suggested change: add shape and dtype validation in `__post_init__` for `TrajectoryMap`, `PoseSequence`, and `SegmentationResult`.

2. `TrajectoryMap.save()` serializes scalar metadata as zero-dimensional arrays. It works, but it is awkward and under-documented. References: `src/motion_mirror/types.py:53-61`. Suggested change: keep the schema documented in one place and validate it on load.

3. `TrajectoryMap.load()` trusts file contents completely. No shape checks, no missing-key handling, no versioning. References: `src/motion_mirror/types.py:63-71`. Suggested change: introduce an explicit file schema version and validate keys and dimensions.

## `src/motion_mirror/hardware.py`

1. `_BACKEND_VRAM` exists and is then ignored. The thresholds are duplicated in `recommend_backend()`. Reference: `src/motion_mirror/hardware.py:49-56`, `src/motion_mirror/hardware.py:87-92`. Suggested change: centralize backend metadata in one table and use it everywhere.

2. `get_gpu_info()` catches every exception and returns `None`. That makes the import-safe story easy, but it also erases real driver/runtime failures. References: `src/motion_mirror/hardware.py:27-46`. Suggested change: catch expected environment failures and log or re-raise unexpected runtime faults.

3. `auto_config()` falls back to `wan-move-14b` when there is no CUDA GPU. That is the worst possible default because it silently selects a backend the machine cannot actually run. References: `src/motion_mirror/hardware.py:128-139`. Suggested change: fail hard, or fall back to `mock`, or require explicit user choice.

4. `auto_config()` rebuilds configs via `base.__slots__`. That is brittle and unnecessary when `dataclasses.replace()` exists. References: `src/motion_mirror/hardware.py:136-147`. Suggested change: use `dataclasses.replace(base, backend=..., offload_model=...)`.

## `src/motion_mirror/pipeline.py`

1. `field` is imported and unused. That is minor, but it matches the general lack of cleanup. Reference: `src/motion_mirror/pipeline.py:3`.

2. The pipeline mutates config objects with `object.__setattr__` to rewrite deprecated backends. That is ugly and avoidable. Reference: `src/motion_mirror/pipeline.py:83-90`. Suggested change: return a replaced config instead of mutating an existing one.

3. The pipeline does not validate the character image for multiple people even though the public API exports `MultipleCharactersError` and the README documents it. References: `src/motion_mirror/pipeline.py:107-114`, `src/motion_mirror/extract/segment.py:21`, `README.md:181-196`. Suggested change: add a dedicated character-image validation stage or remove the claim.

4. Artifact naming is fixed and overwritten on every run: `segmented.png`, `trajectory.npz`, `generated.mp4`, `result.mp4`. That kills traceability and makes debugging multi-run comparisons harder than it needs to be. References: `src/motion_mirror/pipeline.py:115`, `src/motion_mirror/pipeline.py:122`, `src/motion_mirror/pipeline.py:138`. Suggested change: create per-run subdirectories or deterministic run IDs.

5. Backend routing is hard-coded and string-based. That is manageable for two backends and ugly already. References: `src/motion_mirror/pipeline.py:119-132`. Suggested change: route through a backend registry object instead of `if cfg.backend in (...)`.

## `src/motion_mirror/cli.py`

1. `_PRESETS_DIR = Path(__file__).parent.parent.parent / "presets"` is a repo-layout hack, not a packaged application design. Reference: `src/motion_mirror/cli.py:34`. Suggested change: ship presets with the package and load them through package resources.

2. `_MODEL_SPECS` is an untyped dict of dicts. That is lazy design. Reference: `src/motion_mirror/cli.py:49-96`. Suggested change: replace it with a typed `ModelSpec` dataclass.

3. The CLI downloads assets it cannot actually use. `wan-move-fast` and `sam2` are both represented as downloadable and usable, but the runtime path is incomplete or disconnected. References: `src/motion_mirror/cli.py:80-93`, `src/motion_mirror/extract/segment.py:64-69`, `src/motion_mirror/generate/wan_move.py:148-195`. Suggested change: hide unfinished assets from `download` until runtime is real.

4. `run()` accepts raw `str` values for backend, resolution, device, flow estimator, and segmenter, then shoves them into config with no CLI-level validation. References: `src/motion_mirror/cli.py:111-176`. Suggested change: use `typer.Option` enums or callbacks for validation.

5. `run()` prints a configuration summary that can be materially misleading because some flags do nothing and some backends are stubs. References: `src/motion_mirror/cli.py:178-190`. Suggested change: do not echo fake confidence. Reject unsupported combinations early.

6. `download()` treats any non-empty snapshot directory as already cached. That is fragile and wrong. A partial download, stale directory, or interrupted run counts as success. References: `src/motion_mirror/cli.py:263-277`. Suggested change: validate a manifest file, expected filenames, and sizes before declaring a model cached.

7. `download()` stores `expected_bytes` but never validates downloaded size or checksums after download. References: `src/motion_mirror/cli.py:49-96`, `src/motion_mirror/cli.py:225-277`. Suggested change: enforce completeness, not just available disk.

8. `presets()` accepts `list_` and then ignores it. That is dead CLI surface. References: `src/motion_mirror/cli.py:285-317`. Suggested change: remove the unused option or add alternate behavior.

9. `benchmark()` recommends backends that may not actually work because hardware advice is disconnected from runtime truth. Reference: `src/motion_mirror/cli.py:348-358`. Suggested change: only recommend backends whose implementations are actually ready.

## `src/motion_mirror/extract/segment.py`

1. `MultipleCharactersError` is imported and never used. Reference: `src/motion_mirror/extract/segment.py:21`. Suggested change: either implement character-image person-count validation here or stop pretending this file does it.

2. The `TYPE_CHECKING` block does nothing. Reference: `src/motion_mirror/extract/segment.py:24-25`. Suggested change: delete it.

3. `_get_sam2_predictor()` ignores the project's own cache layout. The CLI download command stores SAM-2 under `~/.cache/motion-mirror/sam2`, but runtime loads through `SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")` without using `cfg.model_cache("sam2")`. The advertised download path and runtime path are disconnected. References: `src/motion_mirror/extract/segment.py:49-69`, `src/motion_mirror/cli.py:88-93`. Suggested change: either load from the configured motion-mirror cache or drop the custom download command.

4. `_segment_rembg()` assumes `rembg` is installed and does not provide a clean error message when it is missing. References: `src/motion_mirror/extract/segment.py:123-127`. Suggested change: catch `ImportError` and raise a clear dependency message.

5. `_segment_rembg()` uses `Image.open()` without a context manager. That is not catastrophic, but it is sloppy resource handling. Reference: `src/motion_mirror/extract/segment.py:126`. Suggested change: `with Image.open(...) as image:`.

6. `_segment_sam2()` is built around a center-point prompt assumption. That is a weak heuristic masquerading as a backend. References: `src/motion_mirror/extract/segment.py:152-159`, `src/motion_mirror/extract/segment.py:175-192`. Suggested change: accept explicit prompts or derive them from a real character detector.

7. The CPU inference context is redundant and awkward. On CPU, `inference_ctx` becomes `torch.inference_mode()`, then the code also wraps `with torch.inference_mode(), inference_ctx:`. That is messy and unnecessary. References: `src/motion_mirror/extract/segment.py:180-186`. Suggested change: separate the CPU and CUDA contexts cleanly.

8. The warnings on tiny or huge masks are fine, but there is no fallback behavior after warning. The function still returns whatever mask won the score. References: `src/motion_mirror/extract/segment.py:199-214`. Suggested change: either retry with a different prompt or fail fast when the mask is obviously wrong.

## `src/motion_mirror/extract/pose.py`

1. The file reads the entire video into memory before doing any inference. That is wasteful and unnecessary. References: `src/motion_mirror/extract/pose.py:82-88`, `src/motion_mirror/extract/pose.py:163-241`. Suggested change: stream frames and build outputs incrementally.

2. `frame_count` is read and never used. Reference: `src/motion_mirror/extract/pose.py:70`. Suggested change: either use it for validation or delete it.

3. `warnings` is imported twice. References: `src/motion_mirror/extract/pose.py:49`, `src/motion_mirror/extract/pose.py:74`. Suggested change: keep one import.

4. `use_wholebody` is assigned and never used. Reference: `src/motion_mirror/extract/pose.py:136`, `src/motion_mirror/extract/pose.py:146`. Suggested change: delete it.

5. Person-count validation only happens on frame 0. If frame 0 contains one person and frame 12 contains two, the pipeline happily continues. References: `src/motion_mirror/extract/pose.py:198-237`. Suggested change: enforce person-count consistency across all frames or define a tracking strategy.

6. Subject-size validation only happens on frame 0. That is inconsistent with real motion clips where the subject can enter or leave frame. References: `src/motion_mirror/extract/pose.py:214-237`. Suggested change: validate across the sequence or at least sample multiple frames.

7. `backend_ep = "onnxruntime" if cfg.device == "cuda" else "cpu"` is not obviously correct for rtmlib, and the code gives no reason to trust it. Reference: `src/motion_mirror/extract/pose.py:132`. Suggested change: wrap backend/provider selection in a tested adapter instead of ad hoc strings.

8. The error text mentions `--person-index N` even though the CLI does not support it. Reference: `src/motion_mirror/extract/pose.py:206-210`. Suggested change: do not advertise nonexistent escape hatches.

9. Mock mode generates random keypoints with uniformly high confidence, which makes downstream tests unrealistically forgiving. References: `src/motion_mirror/extract/pose.py:93-106`. Suggested change: use deterministic but semantically structured fixtures for tests.

## `src/motion_mirror/extract/trajectory.py`

1. This file contains the most claimed sophistication and the weakest engineering discipline.

2. `TYPE_CHECKING` is imported and never used. Reference: `src/motion_mirror/extract/trajectory.py:32`. Suggested change: delete it.

3. The function loads all frames into memory and then truncates to match pose length. That is a brute-force pipeline, not a clean one. References: `src/motion_mirror/extract/trajectory.py:103-117`. Suggested change: make trajectory synthesis consume a frame iterator or an already-decoded sequence.

4. Camera stabilization uses `segmentation.mask`, which is the character image mask, not a reference-frame mask. That is a fundamental bug. References: `src/motion_mirror/extract/trajectory.py:119-120`, `src/motion_mirror/extract/trajectory.py:195-240`. Suggested change: derive a reference-video subject ROI independently.

5. `_build_body_transform()` uses all frames' keypoints to compute a reference bbox and then maps them into a fake centered character bbox covering 80% of the image. This is both spec drift and poor geometry. References: `src/motion_mirror/extract/trajectory.py:243-292`. Suggested change: compute the reference bbox from frame 0 or a stable aggregate, and compute the character bbox from actual segmentation or character pose.

6. The transform is anisotropic by default (`sx` and `sy` differ), while the spec explicitly prefers uniform scaling by default. References: `src/motion_mirror/extract/trajectory.py:286-291`. Suggested change: use one scale factor unless the user explicitly requests non-uniform scaling.

7. Layer 1 chooses all keypoints with mean confidence over 0.3 across the full sequence. That can keep bad tracks alive if early frames are good and late frames are garbage. References: `src/motion_mirror/extract/trajectory.py:320-333`. Suggested change: drop or mask per-frame low-confidence points instead of relying on one mean.

8. Layer 2 is a nested Python loop over frames and points. That is slow and unnecessary for NumPy-heavy code. References: `src/motion_mirror/extract/trajectory.py:365-379`. Suggested change: vectorize the Gaussian weighting and displacement application.

9. The implementation hard-codes `n2 = density // 2` and `n3 = density // 4`. That is arbitrary and undocumented in the API. References: `src/motion_mirror/extract/trajectory.py:343`, `src/motion_mirror/extract/trajectory.py:509`. Suggested change: promote layer allocation to a named policy with constants or a config object.

10. `_compute_flow_pair()` catches `ImportError` and then `Exception`, which is the same as just catching everything. It will hide genuine math bugs, shape bugs, and OpenCV faults behind a warning and silently switch estimators. References: `src/motion_mirror/extract/trajectory.py:481-490`. Suggested change: catch only dependency and environment failures, not arbitrary runtime bugs.

11. `_layer3_flow_tracks()` again uses the character-image subject mask in video-space logic. References: `src/motion_mirror/extract/trajectory.py:513-555`. Suggested change: stop reusing cross-space masks.

12. `char_w, char_h = char_size` is assigned and never used. Reference: `src/motion_mirror/extract/trajectory.py:508`. Suggested change: delete it.

13. The fallback zero-flow branch for single-frame input is dead code because the public entrypoint already rejects videos with fewer than two frames. References: `src/motion_mirror/extract/trajectory.py:107-111`, `src/motion_mirror/extract/trajectory.py:558-560`. Suggested change: remove dead branches.

14. The final track composition randomly subsamples from all layers. That means the algorithm can drop skeleton anchors even though those are supposed to be the stable scaffold. References: `src/motion_mirror/extract/trajectory.py:147-153`. Suggested change: preserve Layer 1 anchors first, then fill remaining capacity from layers 2 and 3.

15. There is no temporal resampling to `cfg.num_frames`. The spec talks about mapping reference video frames to a generated frame count, but the implementation just uses however many decoded frames survive truncation. References: `src/motion_mirror/extract/trajectory.py:113-117`, `src/motion_mirror/config.py:33`. Suggested change: add a real resampling stage.

16. There is no camera-stabilization failure warning, even though the spec calls for it. References: `src/motion_mirror/extract/trajectory.py:225-235`. Suggested change: surface a warning when homography estimation fails.

## `src/motion_mirror/generate/models.py`

1. This file is fine as a thin container.

2. The issue is that it is too thin. `GenerationRequest` has no validation, so garbage combinations reach the backend unchecked. References: `src/motion_mirror/generate/models.py:7-24`. Suggested change: validate backend, resolution, frame count, and path existence expectations.

## `src/motion_mirror/generate/controlnet.py`

1. It is a stub. That would be acceptable if the repo treated it like a stub. It does not.

2. `_generate_mock()` does not check `writer.isOpened()`, unlike the mock Wan path. References: `src/motion_mirror/generate/controlnet.py:60-69`, `src/motion_mirror/generate/wan_move.py:88-95`. Suggested change: make the mock helpers consistent.

3. The module-level docstring says v0.1 stub and future v0.2, but the rest of the repo already exposes it as a v0.2a runtime option. References: `src/motion_mirror/generate/controlnet.py:1-5`, `src/motion_mirror/config.py:19-29`, `src/motion_mirror/cli.py:115-129`. Suggested change: stop exposing unfinished internals.

## `src/motion_mirror/generate/wan_move.py`

1. This is the most misleading file in the repo because it looks serious while dodging the actual hard part.

2. The real path never consumes trajectories for conditioning. It reads density and converts it into a prompt. That is not motion control. References: `src/motion_mirror/generate/wan_move.py:233-253`. Suggested change: either wire real conditioning into the model invocation or fail hard until that exists.

3. The returned backend is always `"wan-move-14b"` even if the request backend is `"wan-move-fast"`. Reference: `src/motion_mirror/generate/wan_move.py:280-284`. Suggested change: return the actual backend you ran.

4. `model_dir = config.model_cache("wan-move")` hardcodes the 14B cache regardless of backend. Reference: `src/motion_mirror/generate/wan_move.py:148`. Suggested change: select cache paths from a backend spec registry.

5. If local weights exist but are not in diffusers format, the code falls back to the HF model ID and lets diffusers use its own cache. That makes the custom downloader partly pointless and hard to reason about. References: `src/motion_mirror/generate/wan_move.py:181-195`. Suggested change: either require the expected layout or manage all downloads through one cache strategy.

6. `offload_model` and `t5_cpu` are ignored. Sequential offload is always turned on and there is no T5 placement control. References: `src/motion_mirror/generate/wan_move.py:203-214`, `src/motion_mirror/config.py:36-38`. Suggested change: honor the config or delete the flags.

7. The generator never validates that `config.device` is compatible with the chosen path. It happily creates `torch.Generator(device=config.device)` and assumes that matches the runtime state. Reference: `src/motion_mirror/generate/wan_move.py:243`. Suggested change: validate device compatibility before model load.

8. The negative prompt is hard-coded at module scope with no configurability. Reference: `src/motion_mirror/generate/wan_move.py:119-127`. Suggested change: make prompts explicit configuration, not magic constants.

9. The real path does not test or assert any coupling between trajectory length and generated frames. The trajectory file could be inconsistent and the generator would not care because it is not using it. References: `src/motion_mirror/generate/wan_move.py:233-253`. Suggested change: once motion conditioning exists, enforce frame-count compatibility.

10. The code comments explain why sequential offload is needed, but the implementation never branches on VRAM strategy. The comments are more precise than the code. References: `src/motion_mirror/generate/wan_move.py:209-214`. Suggested change: move VRAM policy into explicit runtime branching.

## `src/motion_mirror/postprocess/audio.py`

1. `static_ffmpeg.add_paths()` runs at module import time. That is a side effect in a library module. Reference: `src/motion_mirror/postprocess/audio.py:16-20`. Suggested change: move PATH mutation into a setup helper or into the function.

2. The probe error message dumps `exc.stderr` without decoding, while mux errors decode it. The error handling is inconsistent. References: `src/motion_mirror/postprocess/audio.py:60-65`, `src/motion_mirror/postprocess/audio.py:94-97`. Suggested change: normalize stderr handling.

3. The audio-present branch is untested in the non-GPU suite. Every existing test uses silent videos, so the only heavily exercised path is the early return. References: `tests/test_audio.py:17-90`. Suggested change: add a real mux-path test with monkeypatched ffmpeg or a tiny synthetic AAC source.

4. Returning `generated_video_path` unchanged when the source has no audio is fine, but it makes `output_path` meaningless in that branch. That should be clearly documented because it is surprising. References: `src/motion_mirror/postprocess/audio.py:71-76`, `tests/test_audio.py:46-55`. Suggested change: document it explicitly or always honor `output_path`.

## `src/motion_mirror/ui/app.py`

1. `create_app(config=None)` is untyped. Reference: `src/motion_mirror/ui/app.py:7`. Suggested change: type it as `MotionMirrorConfig | None`.

2. The UI backend dropdown is stale. It exposes `controlnet` and hides `wan-move-fast` and `wan-1.3b-vace`. References: `src/motion_mirror/ui/app.py:81-85`. Suggested change: keep UI backend choices sourced from the same registry as the CLI.

3. The UI does not surface any v0.2a controls: no `segmenter`, no `flow_estimator`, no `offload_model`, no `t5_cpu`, no auto backend. References: `src/motion_mirror/ui/app.py:23-44`, `src/motion_mirror/cli.py:121-129`. Suggested change: either expose the same supported surface as the CLI or explicitly keep the UI minimal and documented as such.

4. The mock resolution in the UI is `128x64`, while README examples still describe `64x32` and 3 frames for mock. References: `src/motion_mirror/ui/app.py:86-109`, `README.md:143-145`, `presets/mock.toml:5-7`. Suggested change: stop drifting examples and defaults.

5. `on_run()` is defined as a closure and not separately testable. That is why the tests reimplement it instead of testing it. References: `src/motion_mirror/ui/app.py:23-44`, `tests/test_ui.py:52-112`. Suggested change: extract it into a small helper function and test that directly.

## `tests/test_config.py`

1. These are minimal sanity checks. Fine as smoke tests.

2. They do not validate runtime invariants because runtime invariants do not exist yet. Suggested change: once config validation is added, expand these tests to cover rejection paths.

## `tests/test_config_v02a.py`

1. The file mostly proves that strings can be assigned, not that features work. References: `tests/test_config_v02a.py:18-54`. Suggested change: rename these as surface tests, not behavior tests.

2. `test_pipeline_new_backends_in_valid_set()` is weak. It explicitly accepts arbitrary exceptions as long as the exception is not "unknown backend." That is not a meaningful quality assertion. References: `tests/test_config_v02a.py:138-171`. Suggested change: assert concrete expected failures for unfinished backends, or better, remove unfinished backends from the valid set.

3. CLI tests here only assert help text and preset names. They do not prove runtime support. References: `tests/test_config_v02a.py:176-210`. Suggested change: keep them, but do not confuse them with implementation coverage.

## `tests/test_exceptions.py`

1. The hierarchy tests are fine.

2. The integration section is light but acceptable.

3. It still does not test the advertised `MultipleCharactersError` path because that path does not exist. Suggested change: either implement the path or stop exporting/testing the exception as if it were wired in.

## `tests/test_hardware.py`

1. `recommend_backend()` tests are reasonable for pure logic.

2. There is no test for the worst behavior in the module: `auto_config()` on a machine with no GPU falling back to 14B. Suggested change: add that test and then fix the behavior.

## `tests/test_types.py`

1. These are shape tests only. That is fine for now.

2. Once type invariants are added, this file should become the schema guardrail.

## `tests/test_smoke.py`

1. Pure import and container checks. Fine. Not meaningful quality evidence.

## `tests/test_segment.py`

1. These tests rely on whatever `rembg` happens to return for a synthetic solid image, but they only assert shapes and file existence. References: `tests/test_segment.py:21-96`. Suggested change: if segmentation quality matters, assert mask semantics on controlled fixtures, not just shapes.

2. There is no test for missing `rembg` dependency handling. Suggested change: add one once dependency errors are cleaned up.

## `tests/test_segment_sam2.py`

1. The mocked SAM-2 tests are fine for dispatch.

2. The GPU test uses a solid color image and expects a meaningful segmentation result. That is not a stable or realistic fixture. References: `tests/test_segment_sam2.py:203-224`. Suggested change: use a real human cutout fixture or keep this as a smoke-only test.

3. There is still no test proving that the CLI `download --model sam2` artifact is used by runtime, because runtime ignores that cache path today. Suggested change: add one after cache usage is fixed.

## `tests/test_pose.py`

1. Mock path tests are structurally fine.

2. They reinforce the wrong abstraction by relying on random keypoints rather than deterministic semantic tracks. Suggested change: use explicit fixture keypoints that model actual bodies.

## `tests/test_pose_gpu.py`

1. These are smoke tests on synthetic noise videos. They deliberately skip if no meaningful person is detected, which makes sense operationally but tells you almost nothing about pose quality. References: `tests/test_pose_gpu.py:41-82`. Suggested change: use one tiny real-person fixture clip under test assets.

## `tests/test_generate.py`

1. The mock-path tests are fine for output file plumbing.

2. The "real path" tests only assert `NotImplementedError` or missing weights. They do not test the core failure that the real 14B path ignores trajectories. References: `tests/test_generate.py:124-178`. Suggested change: once motion conditioning is real, add assertions that changing trajectories changes the generated condition input.

## `tests/test_generate_gpu.py`

1. These are the most misleading tests in the repo. They prove that a video file can be emitted by the Wan image-to-video pipeline. They do not prove motion transfer because the implementation does not condition on trajectories. References: `tests/test_generate_gpu.py:41-128`, `src/motion_mirror/generate/wan_move.py:233-253`. Suggested change: do not describe these as motion-control validation until the conditioning path exists.

## `tests/test_audio.py`

1. Every test uses silent videos, so the actual mux branch is untested. References: `tests/test_audio.py:17-90`. Suggested change: add at least one audio-present test with a generated tone track or a mocked ffmpeg pipeline.

## `tests/test_ui.py`

1. These tests are weak because the code is hard to test cleanly.

2. `test_on_run_missing_inputs_returns_error()` and `test_on_run_mock_produces_output()` do not call the real `on_run()` callback. They reimplement it. That is not a UI callback test. References: `tests/test_ui.py:52-112`. Suggested change: extract the callback into a named helper and test it directly.

## `tests/test_cli.py`

1. CLI tests are mostly fine for smoke coverage.

2. They do not catch packaging issues because they run from the repo, not an installed wheel. That is why the preset bug survives. References: `tests/test_cli.py:53-70`, `src/motion_mirror/cli.py:34-44`. Suggested change: add a packaging test that installs the wheel in a temp environment and checks `motion-mirror presets --list`.

## `tests/test_pipeline_integration.py`

1. The docstring is overstated. This is not a meaningful primary CI gate for the real pipeline. It is a mock-path integration smoke test. References: `tests/test_pipeline_integration.py:1-6`. Suggested change: rename it accordingly.

2. `test_pipeline_controlnet_backend()` does not test the controlnet path at all. It sets `backend="mock"` and explicitly says so in the comment. References: `tests/test_pipeline_integration.py:137-151`. Suggested change: either route into a patched real controlnet selection path or delete the test.

## `tests/test_trajectory.py`

1. The tests are heavily shape-oriented and mostly use same-size image/video fixtures, which is exactly why the coordinate-space bug survives. References: `tests/test_trajectory.py:102-312`. Suggested change: add mismatched reference/video dimensions and off-center character fixtures.

2. `test_body_transform_maps_to_char_space()` is too weak. It only checks that the transformed point remains in `[0,1]`, which proves almost nothing about correct normalization. References: `tests/test_trajectory.py:247-262`. Suggested change: assert exact mapping behavior against a known transform target.

3. There is no test that Layer 1 anchors survive final downsampling. Suggested change: add one, then fix the compositor.

## `tests/test_trajectory_raft.py`

1. The dispatcher tests are good as fallbacks.

2. The end-to-end synthesize test only asserts that `"raft"` gets threaded into `_compute_flow_pair()`. It does not validate any quality or coordinate behavior. References: `tests/test_trajectory_raft.py:100-154`. Suggested change: after fixing space handling, add targeted behavioral checks on a known synthetic flow field.

## `presets/*.toml`

1. `fast.toml` is misleading because the runtime does not actually provide a distinct LightX2V path. References: `presets/fast.toml:2-10`, `src/motion_mirror/generate/wan_move.py:117-195`. Suggested change: remove the preset or implement the backend.

2. `low-vram.toml` points to a backend that still raises `NotImplementedError`. References: `presets/low-vram.toml:2-10`, `src/motion_mirror/generate/controlnet.py:47-52`. Suggested change: remove the preset or implement the backend.

3. `mock.toml` disagrees with the README mock preset table. The code says `128x64`, `4`, `32`; the README says `64x32`, `3`, `16`. References: `presets/mock.toml:5-7`, `README.md:143-145`. Suggested change: stop letting docs and fixtures drift.

## README / Spec Alignment

1. README claims around public backends, presets, and validated hardware are ahead of implementation. The code is not there yet. References: `README.md:202-238`, `src/motion_mirror/generate/controlnet.py:47-52`, `src/motion_mirror/generate/wan_move.py:233-253`.

2. README documents `MultipleCharactersError`, but no such runtime path exists. References: `README.md:181-193`, `src/motion_mirror/extract/segment.py:21`, `src/motion_mirror/pipeline.py:107-114`.

3. README still says non-GPU CI runs `128 tests`, while the current suite is already `161 passed, 9 deselected` on non-GPU. Reference: `README.md:255`. Suggested change: keep CI/documentation synchronized or stop quoting stale counts.

## Suggested Exact Change Set

1. Stop exposing fake backends. Remove `wan-1.3b-vace`, `wan-move-fast`, `low-vram`, and `fast` from public surfaces until runtime support exists.

2. Split backend metadata into a typed registry. One place should define: public name, minimum VRAM, cache key, download source, implementation status, and generator function.

3. Replace config strings with validated enums or `Literal` plus `__post_init__` checks. Reject bad resolution strings, unsupported devices, invalid frame counts, and impossible combinations early.

4. Replace `object.__setattr__` and `base.__slots__` tricks with `dataclasses.replace()`.

5. Rebuild trajectory synthesis around explicit coordinate spaces:
   `ReferenceVideoSpace`, `CharacterImageSpace`, `BodyTransform`, `ReferenceSubjectMask`.
   The current code is too easy to misuse because raw arrays are passed around without context.

6. Preserve Layer 1 anchors during track composition. Fill remaining density budget from layers 2 and 3 deterministically. Do not randomly sample away the only semantically grounded tracks.

7. Implement real frame-count resampling for pose and trajectories to match `cfg.num_frames`.

8. Either implement actual motion conditioning in `wan_move.py` or stop calling the feature motion transfer. Right now the generator is not honoring the output of the trajectory stage.

9. Connect SAM-2 runtime to the motion-mirror cache path or delete the dedicated download command.

10. Package presets correctly and test the installed wheel, not just editable repo execution.

11. Extract UI callback logic into a named helper so tests can call the real function instead of copying it.

12. Replace synthetic-noise GPU tests with one or two real tiny fixtures. Smoke tests are fine, but they are not evidence of correctness.

## Overall Verdict

The codebase has too much story and not enough discipline.

The architecture is trying to look modular, but the boundaries are soft, the configuration surface is ahead of reality, and the tests are giving management-grade reassurance instead of engineering-grade evidence. The worst part is not that the project is unfinished. The worst part is that unfinished and disconnected pieces are already being presented as if they are integrated features.

The next round of work should not be "add more features." It should be "make the current claims true, or remove them."
