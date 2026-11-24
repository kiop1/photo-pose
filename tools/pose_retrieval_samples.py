"""Download a small research subset of CaptureGuide-Bench and extract COCO-17 poses.

Run with Python 3.12, mediapipe==0.10.21, numpy<2, and pillow.
The benchmark is released for research use. Keep the image cache out of public/.
Its paired scenes were made by removing the person from the original portrait.
"""

import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".pose-local/retrieval/samples"
DATASET = "ShutterMuse/CaptureGuide-Bench"
REVISION = "bd7d513ce8be56796395dd51b8fef17f58ff192e"
SUBJECT_SIDE = "subject-side/paper-benchmark"
COCO_FROM_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
POSE_LANDMARK = (
    mp.tasks.vision.PoseLandmark if hasattr(mp.tasks.vision, "PoseLandmark") else mp.solutions.pose.PoseLandmark
)
MEDIAPIPE_LANDMARK_NAMES = tuple(landmark.name.lower() for landmark in POSE_LANDMARK)
# References that passed the visibility filter in an initial pass over 24 pairs.
SAMPLE_FILENAMES = (
    "92_2026-03-20_4：3_室内 or 探店.png",
    "104_2026-03-20_9：16_室内 or 探店.png",
    "196_2026-03-20_3：4_室内 or 探店.jpg",
    "232_2026-03-20_3：4_风景.jpg",
    "253_2026-03-20_3：4_室内 or 探店.jpg",
    "259_2026-03-20_3：4_风景.jpg",
    "274_2026-03-20_9：16_城市街景.jpg",
    "276_2026-03-20_9：16_室内 or 探店.jpg",
    "284_2026-03-20_9：16_城市街景.jpg",
    "285_2026-03-20_9：16_城市街景.jpg",
    "288_2026-03-20_9：16_城市街景.jpg",
    "303_2026-03-20_3：4_城市街景.jpg",
    "304_2026-03-20_3：4_风景.jpg",
    "312_2026-03-20_3：4_风景.jpg",
    "318_2026-03-20_3：4_风景.jpg",
    "380_2026-03-20_3：4_风景.jpg",
    "538_2026-03-21_9：16_城市街景.jpg",
    "576_2026-03-23_3：4_室内 or 探店.png",
)


def read(url):
    with urlopen(url, timeout=40) as response:
        return response.read()


def prepare_images(filename):
    sample_id = f"cg-{int(filename.split('_', 1)[0]):03d}"
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    sample = {"id": sample_id, "benchmark_filename": filename}
    for kind, directory, source_key in (
        ("scene", SUBJECT_SIDE, "scene_source"),
        ("reference", f"{SUBJECT_SIDE}-gt", "source"),
    ):
        url = f"{base}/{directory}/{quote(filename)}"
        path = OUT / f"{sample_id}-{kind}.jpg"
        if not path.is_file():
            with Image.open(io.BytesIO(read(url))) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((960, 960), Image.Resampling.LANCZOS)
                temp = path.with_suffix(".jpg.partial")
                image.save(temp, format="JPEG", quality=93)
                temp.replace(path)
        sample[kind] = str(path.relative_to(ROOT))
        sample[source_key] = url
    return sample


def joint(point):
    coordinates = (point.x, point.y, point.visibility)
    if not all(math.isfinite(value) for value in coordinates):
        return [0, 0, 0]
    if not 0 <= point.x <= 1 or not 0 <= point.y <= 1:
        return [0, 0, 0]
    return [round(max(0, min(1, value)), 5) for value in coordinates]


def native_landmarks(points):
    return [
        {"name": name, "x": point.x, "y": point.y, "z": point.z,
         "visibility": point.visibility, "presence": point.presence}
        for name, point in zip(MEDIAPIPE_LANDMARK_NAMES, points, strict=True)
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as workers:
        samples = list(workers.map(prepare_images, SAMPLE_FILENAMES))

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "public/pose_landmarker_lite.task")),
        num_poses=1,
    )
    result = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for sample in samples:
            with Image.open(ROOT / sample["reference"]) as reference:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(reference.convert("RGB")))
            poses = landmarker.detect(image).pose_landmarks
            if not poses:
                print(f"{sample['id']}: no pose detected")
                continue
            points = [joint(poses[0][index]) for index in COCO_FROM_MEDIAPIPE]
            visible_body = sum(point[2] >= 0.5 for point in points[5:])
            if visible_body < 8:
                print(f"{sample['id']}: only {visible_body} visible body joints")
                continue
            result.append({
                **sample,
                "points": points,
                "landmarks33": native_landmarks(poses[0]),
                "keypoint_method": "MediaPipe Pose Landmarker Lite",
            })

    if len(result) < 8:
        raise RuntimeError(f"Only {len(result)} usable reference poses")
    manifest = OUT / "manifest.json"
    temp = manifest.with_suffix(".json.partial")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(manifest)
    print(f"Wrote {len(result)} samples to {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
