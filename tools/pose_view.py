"""Describe visible portrait framing using MediaPipe landmarks."""

import math

LEGS = ((23, 25, 27, 31), (24, 26, 28, 32))


def in_frame(point):
    return 0 <= point["x"] <= 1 and 0 <= point["y"] <= 1


def reliable(point, *, visibility=0.5):
    return in_frame(point) and point["visibility"] >= visibility and point["presence"] >= 0.65


def pose_view(landmarks33, size):
    view = {"framing": "unknown", "leg_shape": "unknown", "camera_pitch": "unknown"}
    if landmarks33 is None or len(landmarks33) != 33:
        return view
    if not all(reliable(landmarks33[i]) for i in (0, 11, 12)):
        return view

    angles = []
    for indices in LEGS:
        hip, knee, ankle, toe = (landmarks33[i] for i in indices)
        ankle_visible = reliable(ankle, visibility=0.45)
        if not (reliable(hip) and reliable(knee) and in_frame(toe) and (ankle_visible or reliable(toe))):
            continue
        foot = ankle if ankle_visible else toe
        u, v = [
            ((point["x"] - knee["x"]) * size[0], (point["y"] - knee["y"]) * size[1])
            for point in (hip, foot)
        ]
        length = math.hypot(*u) * math.hypot(*v)
        if length > 0:
            cosine = (u[0] * v[0] + u[1] * v[1]) / length
            angle = math.degrees(math.acos(max(-1, min(1, cosine))))
            angles.append(angle)

    if len(angles) == 2:
        view["framing"] = "full_body"
        # Foreshortening can hide knee flexion, so emit only strong evidence of bending.
        if max(angles) <= 130:
            view["leg_shape"] = "bent"
    elif all(
        math.isfinite(point["x"]) and math.isfinite(point["y"]) and not in_frame(point)
        for point in (landmarks33[i] for i in (27, 28, 31, 32))
    ):
        view["framing"] = "partial_body"
    return view


def view_mismatch(query_view, reference_view):
    query_framing = query_view["framing"]
    reference_framing = reference_view["framing"]
    if query_framing == reference_framing or query_framing == "unknown":
        return None
    if reference_framing == "unknown":
        return "framing_unverified" if query_framing == "full_body" else None
    return "framing_mismatch"


def create_landmarker(model_path):
    import mediapipe as mp

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        num_poses=1,
        output_segmentation_masks=True,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_query_pose(image, landmarker):
    from PIL import Image
    from xhs_pose_samples import detect_pose

    # Stay on the same image scale as gallery detections while retaining normalized coordinates.
    sample = image.copy()
    sample.thumbnail((960, 960), Image.Resampling.LANCZOS)
    estimate = detect_pose(sample, landmarker)
    if estimate is None:
        return None
    pose, background = estimate
    return {
        "points": pose["points"],
        "landmarks33": pose["landmarks33"],
        "view": pose_view(pose["landmarks33"], image.size),
        "background_mask": background.resize(image.size, Image.Resampling.NEAREST),
    }
