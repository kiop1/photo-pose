"""Prepare local Xiaohongshu photos and person masks for the retrieval experiment.

Run with Python 3.12, mediapipe==0.10.21, numpy<2, and pillow.
Put locally collected gallery images in .pose-local/retrieval/xhs/raw/.
"""

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from pose_retrieval_samples import COCO_FROM_MEDIAPIPE, ROOT, joint, native_landmarks


OUT = ROOT / ".pose-local/retrieval/xhs"
SOURCES = {
    "wall": "https://www.xiaohongshu.com/explore/6ac31478000000001203b904",
    "field": "https://www.xiaohongshu.com/explore/6af6ab3e00000000180386c8",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path, rows):
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_photo(image, path):
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=93)
    return str(path.relative_to(ROOT))


def crop_screenshot(source, box, path):
    with Image.open(source) as screenshot:
        if box[2] > screenshot.width or box[3] > screenshot.height:
            raise ValueError(f"Crop {box} exceeds screenshot dimensions {screenshot.size}")
        return save_photo(screenshot.crop(box).convert("RGB"), path)


def prepare_screenshots(held_out, gallery):
    provenance = []
    if held_out:
        for kind, box in (
            ("query", (194, 189, 995, 630)),
            ("reference", (0, 309, 920, 1750)),
        ):
            image = crop_screenshot(held_out, box, OUT / f"held-out/field-03-{kind}.jpg")
            provenance.append({"id": f"field-03-{kind}", "image": image, "crop": box})
    if gallery:
        box = (552, 335, 1664, 1831)
        image = crop_screenshot(gallery, box, OUT / "gallery/field-04-reference.jpg")
        provenance.append({"id": "field-04-reference", "image": image, "crop": box, "source": SOURCES["field"]})
    if provenance:
        write_json(OUT / "screenshot-provenance.json", [
            {**item, "provenance": "Crop of user-provided browser screenshot, held outside retrieval index"}
            for item in provenance
        ])


def detect_pose(photo, landmarker):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(photo))
    result = landmarker.detect(image)
    if not result.pose_landmarks or not result.segmentation_masks:
        return None

    person_probability = result.segmentation_masks[0].numpy_view()
    if person_probability.ndim == 3:
        person_probability = person_probability.squeeze(axis=-1)
    foreground = Image.fromarray(np.where(person_probability >= 0.35, 255, 0).astype(np.uint8))
    if foreground.size != photo.size:
        raise ValueError("Person mask dimensions differ from the photo")
    background = ImageOps.invert(foreground.filter(ImageFilter.MaxFilter(9)))
    background_fraction = float(np.mean(np.asarray(background) == 255))

    points = [joint(result.pose_landmarks[0][index]) for index in COCO_FROM_MEDIAPIPE]
    return {
        "points": points,
        "landmarks33": native_landmarks(result.pose_landmarks[0]),
        "visible_body_joints": sum(point[2] >= 0.5 for point in points[5:]),
        "mask_background_fraction": round(background_fraction, 5),
        "keypoint_method": "MediaPipe Pose Landmarker Lite",
    }, background


def prepare_photo(path, landmarker, source=None):
    group, _, index = path.stem.partition("-")
    if not source and (group not in SOURCES or not index.isascii() or not index.isdigit()):
        raise ValueError(f"Name {path.name} using wall-01 or field-01 followed by its image extension")

    with Image.open(path) as original:
        photo = ImageOps.exif_transpose(original).convert("RGB")
    photo.thumbnail((960, 960), Image.Resampling.LANCZOS)
    estimate = detect_pose(photo, landmarker)
    if estimate is None:
        print(f"{path.stem}: no pose and person mask detected")
        return None
    pose, background = estimate
    visible_body = pose["visible_body_joints"]
    background_fraction = pose["mask_background_fraction"]
    usable_pose = visible_body >= 5

    reference = save_photo(photo, OUT / f"processed/{path.stem}.jpg")
    mask_path = OUT / f"processed/{path.stem}-background.png"
    background.save(mask_path)
    status = "" if usable_pose else " (insufficient visible joints for retrieval)"
    print(f"{path.stem}: {visible_body}/12 visible body joints, {background_fraction:.1%} background{status}")
    return {
        "id": path.stem,
        "group": group,
        "scene": reference,
        "reference": reference,
        "background_mask": str(mask_path.relative_to(ROOT)),
        "source": source or SOURCES[group],
        "source_file": str(path.relative_to(ROOT)),
        "usable_pose": usable_pose,
        **pose,
    }


def prepare_held_out(landmarker):
    path = OUT / "held-out/field-03-reference.jpg"
    if not path.is_file():
        return

    with Image.open(path) as reference:
        estimate = detect_pose(reference.convert("RGB"), landmarker)
    if estimate is None:
        raise RuntimeError(f"No pose and person mask detected in held-out portrait {path.name}")
    pose, background = estimate
    mask_path = path.with_name(f"{path.stem}-background.png")
    background.save(mask_path)
    write_json(path.with_suffix(".json"), {
        "id": path.stem,
        "reference": str(path.relative_to(ROOT)),
        "background_mask": str(mask_path.relative_to(ROOT)),
        **pose,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-screenshot", type=Path)
    parser.add_argument("--gallery-screenshot", type=Path)
    default_catalog = OUT / "blue-field-refs/provenance.json"
    parser.add_argument("--extra-references", type=Path,
                        default=default_catalog if default_catalog.is_file() else None,
                        help="JSON catalog of additional portraits with image, source and optional angle.")
    args = parser.parse_args()

    prepare_screenshots(args.held_out_screenshot, args.gallery_screenshot)
    paths = sorted(path for path in (OUT / "raw").glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())
    if not paths:
        raise SystemExit(f"No gallery images found in {OUT / 'raw'}")
    if len({path.stem for path in paths}) != len(paths):
        raise SystemExit("Input gallery files must have distinct names before the extension")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "public/pose_landmarker_lite.task")),
        num_poses=1,
        output_segmentation_masks=True,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        rows = [row for path in paths if (row := prepare_photo(path, landmarker)) is not None]
        if args.extra_references:
            identifiers = {path.stem for path in paths}
            for ref in json.loads(args.extra_references.read_text()):
                path = ROOT / ref["image"]
                if path.stem in identifiers:
                    raise ValueError(f"Duplicate reference ID: {path.stem}")
                identifiers.add(path.stem)
                if row := prepare_photo(path, landmarker, ref["source"]):
                    row["camera_angle"] = ref.get("angle")
                    rows.append(row)
        prepare_held_out(landmarker)
    if not rows:
        raise SystemExit("No gallery image has pose landmarks and a person mask")
    write_json(OUT / "manifest.json", rows)
    print(f"Wrote {len(rows)} photos to {(OUT / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
