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
- **Use:** Tensor operations and model inference

### onnxruntime / onnxruntime-gpu (optional — GPU path only)
- **License:** MIT
- **Source:** https://github.com/microsoft/onnxruntime
- **Use:** ONNX model inference for DWPose

### rtmlib (optional — GPU path only)
- **License:** Apache 2.0
- **Source:** https://github.com/Tau-J/rtmlib
- **Use:** DWPose-L pose estimation wrapper

---

## Model Weights

### Wan2.1-I2V-14B-720P (primary generation backend)
- **License:** Apache 2.0
- **Source:** https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- **Provider:** Alibaba / Wan-AI
- **Use:** Image-to-video generation backbone

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

---

## Output Licensing

Motion Mirror code is Apache 2.0. The Wan2.1 model weights are also Apache 2.0,
which permits commercial use of both the models and their generated outputs,
subject to applicable laws and Alibaba's acceptable use policy. Users generating
videos commercially should review the full Wan2.1 model card at
https://huggingface.co/Wan-AI for complete terms.

Motion Mirror does not impose any additional restrictions on generated outputs
beyond those of the upstream model licenses.
