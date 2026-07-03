"""Pick and validate smoke inputs for GPU validation.

Runs LOCALLY (CPU). Tries candidate Wikimedia Commons files in order and
gates each with the SAME detector stack the pipeline uses on the pod:
rtmlib Wholebody with the repo's cached DWPose weights (yolox_l.onnx +
dw-ll_ucoco_384.onnx), replicating extract_pose's frame-0 rules exactly
(raw detector person count == 1, keypoint-bbox area fraction >= 5% error /
10% warning thresholds; see src/motion_mirror/extract/pose.py).

A cheaper rtmlib Body prefilter previously passed a clip whose frame 0 had
2 Wholebody detections — the pod then failed with
MultiplePeopleDetectedError. Hence: validate with the real stack only.

The pipeline only accepts .mp4/.mov/.avi/.mkv reference videos, so the pod
trims + transcodes the source to H.264 MP4 starting at ``trim_start_seconds``
for ``trim_seconds``. This script scans candidate start offsets and records
the first window whose frames pass.

Usage:  python runpod-validation/validate_inputs.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "inputs"
DWPOSE_DIR = INPUT_DIR / "dwpose"
SAMPLES_JSON = HERE / "samples.json"

TRIM_SECONDS = 6.0
MIN_BBOX_AREA_FRAC = 0.10  # pipeline: <0.05 error, <0.10 warning — clear both
VIDEO_SAMPLE_FRAMES = 5

DWPOSE_WEIGHTS = {
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
}

VIDEO_CANDIDATES = [
    {
        "title": "Chinese man dancing.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0e/Chinese_man_dancing.webm",
        "license": "CC BY-SA 4.0",
    },
    {
        "title": "Plié, ballet technique tutorial.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/60/Pli%C3%A9%2C_ballet_technique_tutorial.webm",
        "license": "CC BY-SA 3.0",
    },
    {
        "title": "A woman dancing baamaya dance in Northern Ghana.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/98/A_woman_dancing_baamaya_dance_in_Northern_Ghana.webm",
        "license": "CC BY-SA 4.0",
    },
    {
        "title": "An old man dancing.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/11/An_old_man_dancing.webm",
        "license": "CC BY-SA 4.0",
    },
    {
        "title": "A man performing warrior dance in northern Ghana.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/48/A_man_performing_warrior_dance_in_northern_Ghana.webm",
        "license": "CC BY-SA 4.0",
    },
    {
        "title": "B-boy Chris 'Cristyle' Gatdula R16 freestyle.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/25/B-boy_Chris_%27Cristyle%27_Gatdula_R16_freestyle.webm",
        "license": "CC BY-SA 3.0",
    },
]

IMAGE_CANDIDATES = [
    {
        "title": "Ballet dancer in Matosinhos (Unsplash).jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Ballet_dancer_in_Matosinhos_%28Unsplash%29.jpg",
        "license": "CC0",
    },
    {
        "title": "Male dancer practices ballet outdoors beside graffiti wall.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/02/Male_dancer_practices_ballet_outdoors_beside_graffiti_wall.jpg",
        "license": "CC0",
    },
    {
        "title": "Ballet dancer and powder (Unsplash).jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/71/Ballet_dancer_and_powder_%28Unsplash%29.jpg",
        "license": "CC0",
    },
]


def _download(url: str, dest: Path) -> str:
    """Download url to dest (429-aware retry), return sha256 hex digest."""
    if not dest.exists():
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "motion-mirror-validator/1.0"}
                )
                with urllib.request.urlopen(req, timeout=300) as resp, open(
                    dest, "wb"
                ) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 3:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
    return hashlib.sha256(dest.read_bytes()).hexdigest()


_TRACKER = None


def _tracker():
    """rtmlib Wholebody with the repo's own DWPose weights — pipeline-exact."""
    global _TRACKER
    if _TRACKER is None:
        DWPOSE_DIR.mkdir(parents=True, exist_ok=True)
        for name, url in DWPOSE_WEIGHTS.items():
            _download(url, DWPOSE_DIR / name)
        from rtmlib import Wholebody

        _TRACKER = Wholebody(
            det=str(DWPOSE_DIR / "yolox_l.onnx"),
            pose=str(DWPOSE_DIR / "dw-ll_ucoco_384.onnx"),
            to_openpose=False,
            backend="onnxruntime",
            device="cpu",
        )
    return _TRACKER


def _pipeline_frame_check(frame_bgr: np.ndarray) -> tuple[int, float]:
    """Replicate extract_pose frame-0 logic: (raw person count, bbox area frac)."""
    kps_xy, kps_score = _tracker()(frame_bgr)
    if kps_xy is None or len(kps_xy) == 0:
        return 0, 0.0
    kps_xy = np.asarray(kps_xy, dtype=np.float32)
    kps_score = np.asarray(kps_score, dtype=np.float32)
    if kps_xy.ndim == 2:
        kps_xy = kps_xy[np.newaxis]
        kps_score = kps_score[np.newaxis] if kps_score.ndim == 1 else kps_score
    num_people = kps_xy.shape[0]
    conf = kps_score[0]
    xy = kps_xy[0][conf > 0.3]
    frac = 0.0
    if xy.size > 0:
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        h, w = frame_bgr.shape[:2]
        frac = max(0.0, float((x_max - x_min) * (y_max - y_min))) / (w * h)
    return num_people, frac


def _read_all_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _check_video(path: Path) -> tuple[bool, str, float]:
    """Scan 1 s-spaced start offsets; return (ok, note, trim_start)."""
    frames, fps = _read_all_frames(path)
    if not frames:
        return False, "cv2 cannot open/decode", 0.0
    max_start = len(frames) / fps - TRIM_SECONDS
    if max_start < 0:
        return False, "shorter than trim window", 0.0
    last_note = "no window scanned"
    for start_s in np.arange(0.0, max_start + 1e-6, 1.0):
        lo = int(start_s * fps)
        hi = min(int((start_s + TRIM_SECONDS) * fps), len(frames))
        if hi - lo < 17:
            break
        # Frame 0 of the trimmed clip is the hard pipeline gate. ffmpeg's
        # seek may land a frame or two off, so demand the first ~1 s all
        # read as exactly one person.
        guard = range(lo, min(lo + int(fps), hi), max(1, int(fps // 6)))
        spread = np.linspace(lo, hi - 1, VIDEO_SAMPLE_FRAMES).astype(int)
        ok = True
        for j in [*guard, *spread]:
            persons, frac = _pipeline_frame_check(frames[j])
            if persons != 1:
                last_note = f"start={start_s:.0f}s frame {j}: {persons} persons"
                ok = False
                break
            if j == lo and frac < MIN_BBOX_AREA_FRAC:
                last_note = f"start={start_s:.0f}s frame {j}: bbox area {frac:.2f}"
                ok = False
                break
        if ok:
            return (
                True,
                f"pipeline-exact: 1 person, window [{start_s:.0f},"
                f"{start_s + TRIM_SECONDS:.0f}]s, fps={fps:.1f}",
                float(start_s),
            )
        print(f"    window {start_s:.0f}s: {last_note}")
    return False, last_note, 0.0


def _check_image(path: Path) -> tuple[bool, str, float]:
    img = cv2.imread(str(path))
    if img is None:
        return False, "cv2 cannot read", 0.0
    if max(img.shape[:2]) > 1600:
        s = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)
    persons, frac = _pipeline_frame_check(img)
    if persons != 1:
        return False, f"{persons} persons detected", 0.0
    return True, f"1 person, bbox area {frac:.2f}", 0.0


def _pick(candidates, checker, kind: str) -> dict:
    for cand in candidates:
        dest = INPUT_DIR / cand["url"].rsplit("/", 1)[-1]
        print(f"[{kind}] trying {cand['title']} ...", flush=True)
        try:
            sha = _download(cand["url"], dest)
        except Exception as exc:  # noqa: BLE001 - candidate loop
            print(f"    download failed: {exc}")
            continue
        ok, note, trim_start = checker(dest)
        print(f"    {'PASS' if ok else 'FAIL'}: {note}")
        if ok:
            return {
                **cand,
                "sha256": sha,
                "local_file": dest.name,
                "note": note,
                "trim_start_seconds": trim_start,
            }
    print(f"ERROR: no {kind} candidate passed validation", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    video = _pick(VIDEO_CANDIDATES, _check_video, "video")
    image = _pick(IMAGE_CANDIDATES, _check_image, "image")
    samples = {
        "trim_seconds": TRIM_SECONDS,
        "transcode": "h264 mp4 (pipeline accepts .mp4/.mov/.avi/.mkv only)",
        "video": video,
        "image": image,
    }
    SAMPLES_JSON.write_text(json.dumps(samples, indent=2) + "\n")
    print(f"\nwrote {SAMPLES_JSON}")


if __name__ == "__main__":
    main()
