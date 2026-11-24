# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["lancedb>=0.22,<0.35", "transformers>=4.48,<5", "torch>=2.4", "pillow>=10", "numpy>=2,<3", "mediapipe==1.2.3"]
# ///
"""Run from the repository root with uv run tools/pose_retrieval.py."""
import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch
from PIL import Image, ImageDraw, ImageOps
from mediapipe.tasks.python.vision import PoseLandmarksConnections
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from transformers import AutoImageProcessor, AutoModel
from pose_view import create_landmarker, detect_query_pose, pose_view, view_mismatch
from estimate_camera_view import estimate_camera_view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".pose-local/retrieval"
MODEL = "facebook/dinov2-small"
BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def load_photo(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_encoder():
    local_weights = OUT / "dinov2-small"
    location = local_weights if (local_weights / "model.safetensors").is_file() else MODEL
    processor = AutoImageProcessor.from_pretrained(location, use_fast=False)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(location).eval().to(device)
    print(f"Encoder device: {next(model.parameters()).device}", flush=True)
    torch.set_num_threads(min(os.cpu_count() or 1, 4))

    @torch.inference_mode()
    def encode(path, mask_path=None):
        image = load_photo(path)
        mask = Image.new("L", image.size, 255)
        if isinstance(mask_path, Image.Image):
            mask = mask_path.convert("L")
        elif mask_path is not None:
            with Image.open(mask_path) as source_mask:
                mask = source_mask.convert("L")
            if mask.size != image.size:
                raise ValueError(f"Background mask size differs from {path}")
        mean_rgb = tuple(round(channel * 255) for channel in processor.image_mean)
        # Hide the person before self-attention, then pool only background patches.
        image = Image.composite(image, Image.new("RGB", image.size, mean_rgb), mask)
        image = ImageOps.pad(image, (224, 224), color=mean_rgb)
        mask = ImageOps.pad(mask, image.size, color=0)
        grid = image.width // model.config.patch_size
        coverage = np.asarray(mask.resize((grid, grid), Image.Resampling.BOX)).flatten() / 255
        weights = torch.tensor(coverage, dtype=torch.float32, device=device)
        if not weights.any():
            raise ValueError(f"No background visible in {path}")
        inputs = processor(images=image, do_resize=False, do_center_crop=False, return_tensors="pt")
        patches = model(**inputs.to(device)).last_hidden_state[0, 1:]
        pooled = (patches * weights[:, None]).sum(dim=0) / weights.sum()
        return torch.nn.functional.normalize(pooled, dim=0).cpu().tolist()

    return encode


def projection_details(query_size, ref_size, points, anchors=None):
    w, h = query_size
    rw, rh = ref_size
    scale = min(w / rw, h / rh)
    width, height = rw * scale / w, rh * scale / h
    sx, sy, tx, ty = .5, .5, .5, .5
    if anchors:
        sx, sy, tx, ty, scale = anchors
        width, height = rw * scale / w, rh * scale / h
    projected = [[tx + (float(x) - sx) * width,
                  ty + (float(y) - sy) * height, float(v)] for x, y, v in points]

    def cell(joints):
        visible = [point for point in joints if point[2] >= .5]
        torso = [joints[i] for i in (5, 6, 11, 12) if joints[i][2] >= .5]
        center = torso or visible
        if not center:
            return None
        x = sum(point[0] for point in center) / len(center)
        y = sum(point[1] for point in center) / len(center)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return {"row": min(int(y * 3) + 1, 3), "column": min(int(x * 3) + 1, 3)}

    return {"projected_points": projected, "source_grid_cell": cell(points),
            "target_grid_cell": cell(projected)}


def landmark_state(point):
    if not (0 <= point["x"] <= 1 and 0 <= point["y"] <= 1):
        return "outside"
    if min(point["visibility"], point["presence"]) < .5:
        return "estimated"
    return "visible"


def draw_projection(query, landmarks):
    if len(landmarks) != 33:
        raise ValueError("MediaPipe pose overlay requires 33 landmarks")
    base = query.copy()
    base.thumbnail((720, 720))
    w, h = base.size
    xy = [(p["x"] * (w - 1), p["y"] * (h - 1)) for p in landmarks]
    states = [landmark_state(p) for p in landmarks]
    draw = ImageDraw.Draw(base)
    for third in (1 / 3, 2 / 3):
        lines = [((round(w * third), 0), (round(w * third), h)),
                 ((0, round(h * third)), (w, round(h * third)))]
        for line in lines:
            draw.line(line, fill="#263838", width=3)
            draw.line(line, fill="#edf7f4", width=1)
    for edge in PoseLandmarksConnections.POSE_LANDMARKS:
        a, b = edge.start, edge.end
        if states[a] == "visible" and states[b] == "visible":
            draw.line([xy[a], xy[b]], fill="#12261e", width=4)
            draw.line([xy[a], xy[b]], fill="#b6ff84", width=2)
    for state, (x, y) in zip(states, xy):
        if state == "visible":
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill="#f5ffed", outline="#12261e", width=1)
        elif state == "estimated":
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#ffb94e", width=2)
    return base


def select_poses(matches, limit):
    # Compare limb directions in image pixels, ignoring translation and scale.
    selected, signatures = [], []
    for row in matches:
        with Image.open(row["reference"]) as im:
            size = ImageOps.exif_transpose(im).size
        points = np.asarray(row["points"])
        xy = points[:, :2] * size
        signature = {}
        for a, b in BONES:
            if a < 5 or min(points[a, 2], points[b, 2]) < .5:
                continue
            delta = xy[b] - xy[a]
            length = np.linalg.norm(delta)
            if length > 1:
                signature[(a, b)] = delta / length
        similar = False
        for previous in signatures:
            shared = previous.keys() & signature.keys()
            if len(shared) >= 5:
                angle = np.mean([np.arccos(np.clip(signature[k] @ previous[k], -1, 1)) for k in shared])
                if angle < np.deg2rad(15):
                    similar = True
                    break
        if not similar:
            selected.append(row)
            signatures.append(signature)
        if len(selected) == limit:
            break
    return selected


def write_report(report_dir, manifest_size, queries):
    from urllib.parse import urlsplit

    def src(path):
        asset = Path("photos") / (file_hash(Path(path))[:20] + ".jpg")
        destination = report_dir / asset
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            image = load_photo(path)
            image.thumbnail((960, 960))
            image.save(destination, quality=90)
        return asset.as_posix()

    def grid_label(cell):
        if not cell:
            return "—"
        labels = (("左上", "上中", "右上"), ("左中", "正中", "右中"), ("左下", "下中", "右下"))
        return labels[cell["row"] - 1][cell["column"] - 1]

    body = []
    for index, item in enumerate(queries):
        filename = Path(item["query"]).name
        query = html.escape({"field-03-query.jpg": "格桑花海 · 从截图裁出的纯背景",
                             "field-03-reference.jpg": "格桑花海 · 有人画面，检索时屏蔽人物",
                             "scene-bench.jpg": "黄昏长椅 · 跨场景对照"}.get(filename, filename))
        opened = " open" if index == 0 else ""
        body.append(f'<details id="query-{index + 1}" name="queries"{opened}><summary><span>{query}</span>'
                    f'<span class="count">{len(item["matches"])} 个候选</span></summary>')
        if "camera_label" in item:
            body.append(f'<p class="empty">{html.escape(item["camera_label"])} · {html.escape(item["framing_label"])}。{html.escape(item["message"])}</p>')
        if item.get("scene_matches"):
            body.append('<article><h2>相似场景照片</h2><div class="photos">')
            for row in item["scene_matches"]:
                url = row.get("source", "")
                credit = f' <a href="{html.escape(url, quote=True)}">查看来源</a>' if urlsplit(url).scheme in {"https", "http"} else ""
                body.append(f'<figure><figcaption>背景相似度 {row["similarity"]:.3f}{credit}</figcaption>'
                            f'<img src="{src(row["reference"])}" alt="检索到的场景参考照"></figure>')
            body.append('</div></article>')
        if not item["matches"]:
            body.append(f'<article><div class="photos"><figure><figcaption>输入原图</figcaption><img src="{src(item["query"])}" alt="输入原图"></figure>')
            for excluded in ([] if item.get("scene_matches") else item.get("excluded", [])[:2]):
                reason = html.escape(excluded["reason"])
                body.append(f'<figure><figcaption>背景参考 · {reason}</figcaption><img src="{src(excluded["reference"])}" alt="背景相近的参考照片"></figure>')
            body.append('</div></article>')
        for rank, row in enumerate(item["matches"], 1):
            ident = html.escape(row["id"])
            title_id = f"query-{index + 1}-match-{rank}"
            body.append(f'<article aria-labelledby="{title_id}"><header class="match-head">'
                        f'<h2 id="{title_id}">候选 {rank}<span>{ident}</span></h2>'
                        f'<span class="score">余弦相似度 <strong>{row["similarity"]:.3f}</strong></span>'
                        '</header><div class="photos">')
            for path, label, alt in (
                (item["query"], "输入原图", "用于检索的原始场景照片"),
                (row["reference"], "参考姿势", "匹配的参考人物照片"),
                (row["overlay"], "九宫格叠加", "输入照片上的三分线与投影骨架"),
            ):
                body.append(f'<figure><figcaption>{label}</figcaption>'
                            f'<img src="{src(path)}" alt="{alt}" loading="lazy"></figure>')
            body.append('</div><footer>')
            if "target_grid_cell" in row:
                source = grid_label(row.get("source_grid_cell"))
                target = grid_label(row["target_grid_cell"])
                body.append(f'<span>躯干中心格位 <b>{source} → {target}</b> <span class="muted">参考 → 输入</span></span>')
            if "estimated_count" in row:
                body.append(f'<span>33 点骨架 · 实心 {row["visible_count"]} · 空心估计 {row["estimated_count"]} · 画外 {row["outside_count"]}</span>')
            source_url = row.get("source", "")
            if urlsplit(source_url).scheme in {"https", "http"}:
                escaped_url = html.escape(source_url, quote=True)
                body.append(f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">参考图来源 ↗</a>')
            body.append('</footer></article>')
        body.append('</details>')
    (report_dir / "index.html").write_text(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>场景姿势检索</title><link rel="stylesheet" href="/pose_upload.css"><script defer src="/pose_upload.js"></script>
<style>
:root { color-scheme: dark; --text: #edf2ee; --muted: #aebcb5; --border: #33403b; --accent: #b6ff84; }
* { box-sizing: border-box; }
body { margin: 0; background: #111614; color: var(--text); font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1152px; margin: auto; padding: clamp(20px, 4vw, 48px); }
.intro { margin-bottom: 26px; }
h1 { font-size: clamp(28px, 5vw, 46px); line-height: 1.15; margin: 0 0 12px; }
.intro p { color: var(--muted); margin: 4px 0; }
details { background: #18201d; border: 1px solid var(--border); border-radius: 14px; margin: 14px 0; }
summary { cursor: pointer; padding: 16px 20px; display: flex; gap: 12px; align-items: center; list-style: none; font-weight: 650; }
summary::-webkit-details-marker { display: none; }
summary::after { content: ""; width: 8px; height: 8px; flex: 0 0 8px; border: solid var(--accent); border-width: 0 2px 2px 0; transform: rotate(45deg); }
details[open] summary::after { transform: rotate(225deg); }
summary span:first-child { overflow-wrap: anywhere; }
.count { margin-left: auto; white-space: nowrap; color: var(--muted); font-size: .875rem; font-weight: 400; }
article { padding: 18px 20px; border-top: 1px solid var(--border); }
.match-head, footer { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; column-gap: 14px; row-gap: 5px; }
h2 { margin: 0; font-size: 1rem; }
h2 span { font-weight: 400; font-size: .875rem; color: var(--muted); margin-left: 10px; overflow-wrap: anywhere; }
.score { color: var(--muted); font-size: .875rem; }
.score strong { color: var(--accent); margin-left: 4px; font-variant-numeric: tabular-nums; }
.photos { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
figure { margin: 0; min-width: 0; }
figcaption { font-size: .875rem; margin: 0 0 6px; }
img { display: block; width: 100%; aspect-ratio: 4 / 5; max-height: 420px; object-fit: contain; border-radius: 7px; background: #0b0f0d; }
footer { color: var(--muted); font-size: .875rem; }
footer:has(> *) { margin-top: 12px; }
footer b { color: var(--text); }
footer a { color: var(--accent); margin-left: auto; }
.muted { color: var(--muted); margin-left: 4px; }
.empty { color: var(--muted); padding: 0 20px; }
@media (max-width: 560px) {
  main { padding: 18px 12px; }
  summary, article { padding: 12px; }
  .photos { grid-template-columns: 1fr; gap: 14px; }
  img { max-height: 360px; }
}
</style></head><body><main><header class="intro"><h1>场景姿势检索</h1>
"""
        f'<p>{manifest_size} 个参考样本 · DINOv2-Small · LanceDB</p>'
        '<p>余弦相似度越高，场景向量越接近。按机位和人物景别筛选后叠加骨架。 <a href="matches.json">查看 33 点坐标和得分</a></p></header><section id="pose-upload"></section>'
        + ''.join(body) + '</main></body></html>',
        encoding="utf-8")


def build_index(samples, output_dir, encode):
    table_rows = []
    for sample in samples:
        scene, reference = [ROOT / sample[key] for key in ("scene", "reference")]
        points = np.asarray(sample["points"], dtype=float)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            raise ValueError(f'Invalid COCO-17 points for {sample["id"]}')
        native = sample["landmarks33"]
        if len(native) != 33 or not all(np.isfinite([p[k] for k in ("x", "y", "z", "visibility", "presence")]).all() for p in native):
            raise ValueError(f'Invalid MediaPipe landmarks for {sample["id"]}')
        mask = ROOT / sample["background_mask"] if sample.get("background_mask") else None
        table_rows.append({
            "id": sample["id"], "scene": str(scene), "reference": str(reference),
            "source": sample["source"], "annotated_angle": sample.get("camera_angle"), "scene_sha256": file_hash(scene),
            "ref_sha256": file_hash(reference), "points": points.tolist(), "landmarks33": native,
            "background_mask": str(mask) if mask else None, "usable_pose": sample.get("usable_pose", True),
            "vector": encode(scene, mask),
        })
    db = lancedb.connect(output_dir / "lancedb")
    landmark_type = pa.struct([("name", pa.string())] + [(k, pa.float32()) for k in
                              ("x", "y", "z", "visibility", "presence")])
    schema = pa.schema([(key, pa.string()) for key in
                        ("id", "scene", "reference", "source", "annotated_angle", "scene_sha256", "ref_sha256", "background_mask")] +
                       [("usable_pose", pa.bool_()), ("points", pa.list_(pa.list_(pa.float32(), 3), 17)),
                        ("landmarks33", pa.list_(landmark_type)), ("vector", pa.list_(pa.float32(), 384))])
    table = db.create_table("pose_scene_experiment", table_rows, schema=schema, mode="overwrite")
    return table


CAMERA_LABELS = {"high": "俯拍", "level": "平拍", "low": "仰拍", "unknown": "机位待确认"}
FRAMING_LABELS = {"full_body": "全身入镜", "partial_body": "局部入镜", "unknown": "景别待确认"}


def shoulder_alignment(reference, target, ref_size, target_size):
    def shoulders(points, size):
        a, b = points[5:7]
        if min(a[2], b[2]) < .5:
            return None
        span = float(np.linalg.norm((np.asarray(a[:2]) - b[:2]) * size))
        if span < 4:
            return None
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, span

    src, dst = shoulders(reference, ref_size), shoulders(target, target_size)
    if src and dst:
        return (*src[:2], *dst[:2], dst[2] / src[2])


class RetrievalSession:
    def __init__(self, output_dir, *, encode=None):
        self.table = lancedb.connect(Path(output_dir) / "lancedb").open_table("pose_scene_experiment")
        self.encode = encode or scene_encoder()
        self.detector = create_landmarker(ROOT / "public/pose_landmarker_lite.task")
        self.annotations = json.loads(Path(__file__).with_name("pose_camera_views.json").read_text())["by_sha256"]
        self.views = {}

    def camera(self, path, fingerprint, angle="auto"):
        if angle != "auto":
            return {"angle": angle, "method": "用户指定"}
        reviewed = self.annotations.get(fingerprint)
        if reviewed:
            return {"angle": reviewed["angle"], "method": "人工标注", "evidence": reviewed["evidence"]}
        if fingerprint not in self.views:
            prediction = estimate_camera_view(path)
            value = {"angle": prediction["angle"], "method": "模型估计", "evidence": prediction["evidence"]}
            if prediction.get("error"):
                return {"angle": "unknown", "method": "估计服务未就绪"}
            if prediction["confidence"] < .70:
                value["angle"] = "unknown"
            if len(self.views) >= 256:
                self.views.pop(next(iter(self.views)))
            self.views[fingerprint] = value
        return self.views[fingerprint]

    def search(self, path, report_dir, index=1, top_k=3, angle="auto"):
        start = time.perf_counter()
        query = Path(path).resolve()
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        image = load_photo(query)
        fingerprint = file_hash(query)
        camera = self.camera(query, fingerprint, angle)
        subject = detect_query_pose(image, self.detector)
        view = subject["view"] if subject else {"framing": "unknown"}
        vector = self.encode(query, subject["background_mask"] if subject else None)
        encode_ms = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        rows = (self.table.search(vector).distance_type("cosine")
                .where(f"scene_sha256 != '{fingerprint}' AND ref_sha256 != '{fingerprint}'")
                .limit(self.table.count_rows()).to_list())
        search_ms = 1000 * (time.perf_counter() - start)
        scene_matches = [{"id": row["id"], "reference": row["reference"], "source": row["source"],
                          "similarity": 1 - row["_distance"]} for row in rows[:top_k]]
        eligible, excluded = [], []
        for row in rows:
            if not row["usable_pose"]:
                excluded.append({"id": row["id"], "reference": row["reference"], "reason": "参考照中清楚可见的关节不足",
                                 "similarity": 1 - row["_distance"]})
                continue
            ref_angle = row.get("annotated_angle") or self.camera(row["reference"], row["ref_sha256"])["angle"]
            with Image.open(row["reference"]) as source:
                ref_size = ImageOps.exif_transpose(source).size
            ref_view = pose_view(row["landmarks33"], ref_size)
            reasons = []
            tentative = subject is None and camera["angle"] == "unknown" and ref_angle != "unknown"
            if "unknown" in (camera["angle"], ref_angle) and not tentative:
                reasons.append("机位待确认")
            elif not tentative and camera["angle"] != ref_angle:
                reasons.append(f'机位不同（参考为{CAMERA_LABELS[ref_angle]}）')
            if view_mismatch(view, ref_view):
                reasons.append(f'景别不匹配（参考为{FRAMING_LABELS[ref_view["framing"]]}）')
            anchors = shoulder_alignment(row["points"], subject["points"], ref_size, image.size) if subject else None
            if subject and not anchors:
                reasons.append("双肩定位不足")
            if reasons:
                excluded.append({"id": row["id"], "reference": row["reference"], "reason": "、".join(reasons),
                                 "similarity": 1 - row["_distance"]})
            else:
                eligible.append({**row, "anchors": anchors, "ref_size": ref_size})
        retrieved = []
        for rank, row in enumerate(select_poses(eligible, top_k), 1):
            overlay = report_dir / f"query-{index}-rank-{rank}.jpg"
            options = image.size, row["ref_size"]
            projection = projection_details(*options, row["points"], row["anchors"])
            native = row["landmarks33"]
            mapped = projection_details(*options, [(p["x"], p["y"], p["visibility"]) for p in native],
                                        row["anchors"])["projected_points"]
            grid = [{**p, "x": xy[0], "y": xy[1]} for p, xy in zip(native, mapped)]
            draw_projection(image, grid).save(overlay, quality=92)
            states = [landmark_state(p) for p in grid]
            retrieved.append({**{key: row[key] for key in ("id", "scene", "reference", "source", "landmarks33")},
                              **projection, "projected_landmarks33": grid,
                              "visible_count": states.count("visible"), "estimated_count": states.count("estimated"),
                              "outside_count": states.count("outside"),
                              "similarity": 1 - row["_distance"], "overlay": str(overlay)})
        if camera["angle"] == "unknown" and retrieved:
            message = "空景机位待确认。骨架按参考照片的构图试摆，站位和人物大小可现场调整。"
        elif camera["angle"] == "unknown":
            message = "当前机位待确认。可以手动选择拍摄角度后重新检索。"
        elif not retrieved:
            message = "图库中没有同时满足当前机位和可见身体范围的姿势。"
        elif subject:
            message = "骨架按当前人物的双肩位置和宽度对齐。"
        else:
            message = "未检测到人物。骨架沿用参考构图，站位和支撑面需要现场确认。"
        return {"query": str(query), "camera": camera, "camera_label": CAMERA_LABELS[camera["angle"]] + " · " + camera["method"],
                "framing_label": FRAMING_LABELS[view["framing"]] if subject else "未检测到人物",
                "message": message, "scene_matches": scene_matches, "excluded": excluded, "raw_ranking": [r["id"] for r in rows],
                "encode_ms": round(encode_ms, 1), "search_ms": round(search_ms, 1), "matches": retrieved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUT / "samples/manifest.json")
    parser.add_argument("--query", type=Path, nargs="+", default=[
        ROOT / "public/field-cases/scene-bench.jpg", ROOT / "public/field-cases/scene-alley.jpg"])
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    samples = json.loads(args.manifest.read_text())
    if not samples:
        parser.error("empty reference library")
    encode = scene_encoder()
    build_index(samples, args.output_dir, encode)
    session = RetrievalSession(args.output_dir, encode=encode)
    report_dir = args.output_dir.resolve() / "report"
    result = [session.search(query, report_dir, index=i, top_k=args.top_k)
              for i, query in enumerate(args.query, 1)]
    (report_dir / "matches.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    write_report(report_dir, len(samples), result)
    print(report_dir / "index.html")


if __name__ == "__main__":
    main()
