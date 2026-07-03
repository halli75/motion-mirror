"""Pick and validate CC0 smoke inputs for GPU validation.

Runs LOCALLY (CPU). Tries candidate Wikimedia Commons files in order,
gates each with DWPose (rtmlib, CPU) for exactly-one-person detection,
and writes samples.json recording the chosen URLs, sha256 checksums,
licenses, and the trim/transcode spec the pod must apply.

The pipeline only accepts .mp4/.mov/.avi/.mkv reference videos, so the
pod transcodes the source file to H.264 MP4 (first ``trim_seconds``
seconds) before running the smoke matrix. Validation here samples frames
from the same trimmed window of the original download.

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
SAMPLES_JSON = HERE / "samples.json"

TRIM_SECONDS = 6.0
MIN_BBOX_HEIGHT_FRAC = 0.35
VIDEO_SAMPLE_FRAMES = 5

VIDEO_CANDIDATES = [
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
        "title": "Chinese man dancing.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0e/Chinese_man_dancing.webm",
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
                with urllib.request.urlopen(req, timeout=180) as resp, open(
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


_BODY = None


def _detector():
    global _BODY
    if _BODY is None:
        from rtmlib import Body

        _BODY = Body(mode="balanced", backend="onnxruntime", device="cpu")
    return _BODY


def _person_check(frame_bgr: np.ndarray) -> tuple[int, float]:
    """Return (num_confident_persons, tallest_bbox_height_fraction)."""
    keypoints, scores = _detector()(frame_bgr)
    h = frame_bgr.shape[0]
    persons = 0
    best_frac = 0.0
    for kp, sc in zip(keypoints, scores):
        good = sc > 0.3
        if good.sum() < 5:  # too few confident keypoints -> not a person
            continue
        persons += 1
        ys = kp[good, 1]
        best_frac = max(best_frac, float(ys.max() - ys.min()) / h)
    return persons, best_frac


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
    """Try trim windows starting at 0/3/6/9 s; return (ok, note, trim_start)."""
    frames, fps = _read_all_frames(path)
    if not frames:
        return False, "cv2 cannot open/decode", 0.0
    last_note = ""
    for start_s in (0.0, 3.0, 6.0, 9.0):
        lo = int(start_s * fps)
        hi = int((start_s + TRIM_SECONDS) * fps)
        if hi > len(frames) or hi - lo < 17:
            break
        idxs = np.linspace(lo, hi - 1, VIDEO_SAMPLE_FRAMES).astype(int)
        ok = True
        for j in idxs:
            persons, frac = _person_check(frames[j])
            if persons != 1:
                last_note = f"start={start_s}s frame {j}: {persons} persons"
                ok = False
                break
            if frac < MIN_BBOX_HEIGHT_FRAC:
                last_note = f"start={start_s}s frame {j}: bbox height {frac:.2f}"
                ok = False
                break
        if ok:
            return (
                True,
                f"1 person on {VIDEO_SAMPLE_FRAMES} frames in [{start_s},"
                f"{start_s + TRIM_SECONDS}]s, fps={fps:.1f}",
                start_s,
            )
    return False, last_note, 0.0


def _check_image(path: Path) -> tuple[bool, str, float]:
    img = cv2.imread(str(path))
    if img is None:
        return False, "cv2 cannot read", 0.0
    # Downscale very large images for detection speed; fractions are scale-free.
    if max(img.shape[:2]) > 1600:
        s = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)
    persons, frac = _person_check(img)
    if persons != 1:
        return False, f"{persons} persons detected", 0.0
    if frac < MIN_BBOX_HEIGHT_FRAC:
        return False, f"bbox height {frac:.2f} < {MIN_BBOX_HEIGHT_FRAC}", 0.0
    return True, f"1 person, bbox height {frac:.2f}", 0.0


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
