# Third-Party Licenses

Motion Mirror is Apache 2.0 licensed. The following third-party components are
used at runtime or during model inference. Each component retains its original
license.

---

## Python Runtime Dependencies

### gradio
- **License:** Apache 2.0
- **Source:** https://github.com/gradio-app/gradio
- **Use:** Web UI framework

### typer
- **License:** MIT
- **Source:** https://github.com/tiangolo/typer
- **Use:** CLI framework

### opencv-python
- **License:** Apache 2.0 (Python bindings); LGPL 2.1 (underlying OpenCV library)
- **Source:** https://github.com/opencv/opencv-python
- **Use:** Video I/O, optical flow (Farneback), homography estimation (ORB)

### numpy
- **License:** BSD 3-Clause
- **Source:** https://github.com/numpy/numpy
- **Use:** Array operations throughout the pipeline

### rembg
- **License:** MIT
- **Source:** https://github.com/danielgatis/rembg
- **Use:** Background removal (character segmentation)

### Pillow (PIL)
- **License:** Historical Permission Notice and Disclaimer (HPND) — effectively MIT-compatible
- **Source:** https://github.com/python-pillow/Pillow
- **Use:** Image I/O and compositing

### static-ffmpeg
- **License:** LGPL 2.1 (bundled ffmpeg binary)
- **Source:** https://github.com/zackees/static-ffmpeg
- **Use:** Bundled ffmpeg binary for audio extraction and muxing

### ffmpeg-python
- **License:** Apache 2.0
- **Source:** https://github.com/kkroening/ffmpeg-python
- **Use:** Python wrapper for ffmpeg subprocess calls

### huggingface-hub
- **License:** Apache 2.0
- **Source:** https://github.com/huggingface/huggingface_hub
- **Use:** Model weight download and caching

### rich
- **License:** MIT
- **Source:** https://github.com/Textualize/rich
- **Use:** Terminal output formatting in the CLI

### diffusers (optional — GPU path only)
- **License:** Apache 2.0
- **Source:** https://github.com/huggingface/diffusers
- **Use:** `WanImageToVideoPipeline` for Wan2.1-I2V-14B video generation

### transformers (optional — GPU path only)
- **License:** Apache 2.0
- **Source:** https://github.com/huggingface/transformers
- **Use:** `CLIPVisionModel` image encoder for the Wan pipeline

### accelerate (optional — GPU path only)
- **License:** Apache 2.0
- **Source:** https://github.com/huggingface/accelerate
- **Use:** Model loading and device placement utilities

### torch / torchvision (optional — GPU path only)
- **License:** BSD 3-Clause
- **Source:** https://github.com/pytorch/pytorch / https://github.com/pytorch/vision
- **Use:** Tensor operations, model inference, and optional RAFT optical flow

### onnxruntime / onnxruntime-gpu (optional — GPU path only)
- **License:** MIT
- **Source:** https://github.com/microsoft/onnxruntime
- **Use:** ONNX model inference for DWPose

### rtmlib (optional — GPU path only)
- **License:** Apache 2.0
- **Source:** https://github.com/Tau-J/rtmlib
- **Use:** DWPose-L pose estimation wrapper

### lightx2v (optional — v0.2a fast backend)
- **License:** Apache 2.0
- **Source:** https://github.com/ModelTC/LightX2V
- **Use:** LightX2V runtime for `wan-move-fast` 4-step Wan2.1 inference

### sam2 (optional — v0.2a segmentation/masking)
- **License:** Apache 2.0
- **Source:** https://github.com/facebookresearch/sam2
- **Use:** SAM-2 character segmentation and optional reference-video masking

### gguf (optional — v0.2a quantized backend)
- **License:** MIT
- **Source:** https://github.com/ggml-org/ggml
- **Use:** Loading GGUF-quantized Wan transformer checkpoints through Diffusers

### DiffSynth-Studio / DiffSynth (optional — v0.2b identity backend)
- **License:** Apache 2.0
- **Source:** https://github.com/modelscope/DiffSynth-Studio
- **Use:** Experimental Concat-ID Wan runtime integration for `wan-1.3b-concat-id`

---

## Model Weights

### Wan2.1-I2V-14B-720P (primary generation backend)
- **License:** Apache 2.0
- **Source:** https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- **Provider:** Alibaba / Wan-AI
- **Use:** Image-to-video generation backbone for the current `wan-move-14b` path. True Wan-Move trajectory-guidance weights are tracked separately.

### Wan-Move-14B-480P (planned trajectory-guidance backend)
- **License:** Apache 2.0
- **Source:** https://huggingface.co/Ruihang/Wan-Move-14B-480P
- **Provider:** Alibaba Tongyi Lab / Ruihang Chu
- **Use:** Planned source for true `wan.WanMove` latent trajectory guidance integration.

### Wan2.1-I2V-14B GGUF quantizations
- **License:** Apache 2.0 inherited from upstream Wan2.1 weights unless a specific quantized checkpoint states otherwise
- **Source:** https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf
- **Provider:** city96 / upstream Wan-AI
- **Use:** Experimental GGUF-quantized transformer backend for `wan-move-gguf`

### Wan2.1-I2V LightX2V distilled weights
- **License:** Apache 2.0 inherited from upstream Wan2.1 / LightX2V release terms unless a specific checkpoint states otherwise
- **Source:** https://huggingface.co/lightx2v/Wan2.1-Distill-Models
- **Provider:** ModelTC / LightX2V
- **Use:** Distilled 4-step I2V weights for `wan-move-fast`

### Wan2.1-VACE-1.3B
- **License:** Apache 2.0
- **Source:** https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers
- **Provider:** Alibaba / Wan-AI
- **Use:** Low-VRAM VACE generation backend for `wan-1.3b-vace`

### Wan2.1-T2V-1.3B
- **License:** Apache 2.0
- **Source:** https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- **Provider:** Alibaba / Wan-AI
- **Use:** Base model for the experimental `wan-1.3b-concat-id` identity backend

### Concat-ID-Wan
- **License:** Review the upstream model card before redistribution or commercial use
- **Source:** https://huggingface.co/yongzhong/Concat-ID-Wan
- **Provider:** ML-GSAI / Yong Zhong
- **Use:** Experimental identity adapter weights for `wan-1.3b-concat-id`

### UMT5-XXL Text Encoder
- **License:** Apache 2.0
- **Source:** Bundled within the Wan2.1 checkpoint
- **Use:** Text conditioning for video generation

### Wan VAE
- **License:** Apache 2.0
- **Source:** Bundled within the Wan2.1 checkpoint
- **Use:** Video latent encoding/decoding

### CLIP Vision Encoder (openai/clip-vit-large-patch14 or equivalent)
- **License:** MIT
- **Source:** Bundled within the Wan2.1-I2V-14B-720P-Diffusers checkpoint
- **Use:** Image conditioning for I2V generation

### DWPose-L (pose estimation)
- **License:** Apache 2.0
- **Source:** https://huggingface.co/yzd-v/DWPose
- **Provider:** Tau-J / yzd-v
- **Use:** Whole-body keypoint detection (133 COCO-WholeBody keypoints)

### YOLOX Person Detector (bundled with DWPose)
- **License:** Apache 2.0
- **Source:** https://huggingface.co/yzd-v/DWPose
- **Use:** Person detection prior to pose estimation

### U²-Net (rembg segmentation model)
- **License:** Apache 2.0
- **Source:** https://github.com/xuebinqin/U-2-Net (weights distributed via rembg)
- **Use:** Salient object detection for background removal

### SAM-2 Hiera Large
- **License:** Apache 2.0
- **Source:** https://huggingface.co/facebook/sam2-hiera-large
- **Provider:** Meta / Facebook Research
- **Use:** Optional SAM-2 segmentation and reference masking via `--segmenter sam2` and `--reference-masker sam2`

---

## Output Licensing

Motion Mirror code is Apache 2.0. The Wan2.1 model weights are also Apache 2.0,
which permits commercial use of both the models and their generated outputs,
subject to applicable laws and Alibaba's acceptable use policy. Users generating
videos commercially should review the full Wan2.1 model card at
https://huggingface.co/Wan-AI for complete terms.

Motion Mirror does not impose any additional restrictions on generated outputs
beyond those of the upstream model licenses.
