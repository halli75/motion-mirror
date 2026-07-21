"""Model download specs and fast-mode artifacts, shared by the CLI and backend.

Imports nothing from cli/generate so both can import it without a cycle.
"""
from __future__ import annotations

MODEL_SPECS: dict[str, dict] = {
    "dwpose-pose": {
        "repo_id": "yzd-v/DWPose",
        "filename": "dw-ll_ucoco_384.onnx",
        "expected_bytes": 134_000_000,
        "cache_subdir": "dwpose",
        "label": "DWPose wholebody pose model",
    },
    "dwpose-det": {
        "repo_id": "yzd-v/DWPose",
        "filename": "yolox_l.onnx",
        "expected_bytes": 217_000_000,
        "cache_subdir": "dwpose",
        "label": "DWPose YOLOX detector",
    },
    "wan-1.3b-vace": {
        "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "filename": None,
        "expected_bytes": 19_000_000_000,
        "min_cached_bytes": 3_000_000_000,
        "cache_subdir": "wan-1.3b-vace",
        "label": "Wan2.1-VACE-1.3B (lightweight, ~19 GB download, needs ~9 GB VRAM) [backend: wan-1.3b-vace]",
    },
    "sam2": {
        "repo_id": "facebook/sam2-hiera-large",
        "filename": None,
        "expected_bytes": 1_800_000_000,
        "min_cached_bytes": 500_000_000,
        "cache_subdir": "sam2",
        "label": "SAM-2 Large image segmenter (~1.8 GB) [--segmenter sam2]",
    },
    "wan-14b-vace": {
        "repo_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "filename": None,
        "expected_bytes": 76_000_000_000,
        "min_cached_bytes": 60_000_000_000,
        "cache_subdir": "wan-14b-vace",
        "label": "Wan2.1-VACE-14B full (~75 GB, GPU-unvalidated, explicit --backend only) [backend: wan-14b-vace]",
    },
    "wan-14b-vace-gguf": {
        "repo_id": "QuantStack/Wan2.1_14B_VACE-GGUF",
        "filename": "Wan2.1_14B_VACE-Q4_K_M.gguf",
        "expected_bytes": 11_600_000_000,
        "cache_subdir": "wan-14b-vace-gguf",
        "label": "Wan2.1-VACE-14B Q4_K_M GGUF transformer (~11.6 GB) [backend: wan-14b-vace-gguf]",
    },
    "wan-fast-14b-lora": {
        "repo_id": "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v",
        "filename": "loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        "expected_bytes": 631_344_264,
        "cache_subdir": "wan-fast-14b-lora",
        "label": "LightX2V 14B step/CFG distill LoRA (~631 MB, Apache-2.0) [--fast on wan-14b-vace]",
    },
    "wan-14b-vace-fusionx-gguf": {
        "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF",
        "filename": "Wan2.1_T2V_14B_FusionX_VACE-Q4_K_M.gguf",
        "expected_bytes": 11_629_623_072,
        "cache_subdir": "wan-14b-vace-fusionx-gguf",
        "label": "FusionX pre-merged distilled VACE GGUF Q4_K_M (~11.6 GB, Apache-2.0, EXPERIMENTAL) [--fast on wan-14b-vace-gguf]",
        "experimental_warning": (
            "FusionX GGUF fast mode is EXPERIMENTAL: untested via diffusers; "
            "validated in ComfyUI only."
        ),
    },
    "wan-fast-1.3b": {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "LoRAs/Wan2_1_self_forcing_1_3B/Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors",
        "expected_bytes": 91_233_416,
        "cache_subdir": "wan-fast-1.3b",
        "label": "Self-Forcing DMD 1.3B distill LoRA (~91 MB, NON-COMMERCIAL) [--fast on wan-1.3b-vace]",
        "license_warning": (
            "NON-COMMERCIAL ONLY: these 1.3B fast weights derive from "
            "Self-Forcing (CC-BY-NC-SA-4.0). Generated outputs may not be "
            "used commercially. Motion Mirror code itself remains MIT.\n"
            "Source: https://huggingface.co/Kijai/WanVideo_comfy "
            "(upstream: https://github.com/guandeh17/Self-Forcing)"
        ),
    },
    "wan-14b-vace-base": {
        "repo_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "filename": None,
        "expected_bytes": 12_000_000_000,
        "min_cached_bytes": 9_000_000_000,
        "cache_subdir": "wan-14b-vace-base",
        "allow_patterns": ["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "model_index.json"],
        "label": "Wan2.1-VACE-14B base components sans transformer (~12 GB, for the GGUF backend)",
    },
}

MODEL_GROUPS = {
    "dwpose": ["dwpose-pose", "dwpose-det"],
    "vace": ["wan-1.3b-vace"],
    "vace-14b": ["wan-14b-vace"],
    "vace-14b-gguf": ["wan-14b-vace-gguf", "wan-14b-vace-base"],
    "extras": ["sam2"],
    # Apache-only: the NC-licensed 1.3B artifact must be named explicitly.
    "fast": ["wan-fast-14b-lora"],
    # Validated ~6 GB lineup only; the 14B backends are named explicitly.
    "all": [
        "dwpose-pose",
        "dwpose-det",
        "wan-1.3b-vace",
        "sam2",
    ],
}

# Distill artifact per backend; `artifact` names the MODEL_SPECS entry.
# `gguf_swap` entries replace the quantized transformer (LoRA-on-GGUF is
# unsupported in diffusers); the rest fuse a LoRA.
FAST_BACKEND_SPECS: dict[str, dict] = {
    "wan-14b-vace": {"artifact": "wan-fast-14b-lora", "steps": 4},
    "wan-1.3b-vace": {"artifact": "wan-fast-1.3b", "steps": 4},
    "wan-14b-vace-gguf": {
        "artifact": "wan-14b-vace-fusionx-gguf",
        "steps": 8,
        "gguf_swap": True,
    },
}

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
FAST_GUIDANCE_SCALE = 1.0  # CFG off - distilled students are CFG-free
FAST_FLOW_SHIFT = 5.0
