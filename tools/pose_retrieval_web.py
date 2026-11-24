# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "lancedb>=0.22,<0.35", "transformers>=4.48,<5", "torch>=2.4", "pillow>=10",
#   "numpy>=2,<3", "mediapipe==1.2.3",
# ]
# ///
"""Serve the local pose retrieval report and single-photo search."""

import argparse
import json
import mimetypes
import re
import shutil
import signal
import sys
import tempfile
import threading
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from PIL import Image, ImageOps

TOOLS = Path(__file__).resolve().parent
DEFAULT_OUTPUT = TOOLS.parent / ".pose-local/retrieval/xhs"
MAX_UPLOAD = 64 * 1024 * 1024
MAX_PIXELS = 80_000_000
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
UPLOAD_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
ANGLES = {"auto", "high", "level", "low"}
ASSET_ROUTE = re.compile(r"/uploads/([0-9a-f]{32})/(query|(?:reference|overlay|excluded|scene)-[1-3])\.jpg")
REPORT_EXTENSIONS = {".html", ".jpg", ".jpeg", ".png", ".webp", ".json"}
FILE_FIELDS = {"query", "reference", "scene", "overlay", "background_mask"}


def save_preview(source, destination):
    with Image.open(source) as image:
        preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail((960, 960))
        preview.save(destination, quality=90)


def omit_file_paths(value):
    if isinstance(value, dict):
        return {key: omit_file_paths(item) for key, item in value.items() if key not in FILE_FIELDS}
    if isinstance(value, list):
        return [omit_file_paths(item) for item in value]
    return value


class RetrievalServer(ThreadingHTTPServer):
    def __init__(self, address, output_dir, workspace):
        self.output_dir = output_dir
        self.report_dir = (output_dir / "report").resolve()
        self.workspace = workspace
        self.recent_uploads = deque()
        self.model_lock = threading.Lock()
        self.session = None
        super().__init__(address, RetrievalHandler)

    def search(self, upload, angle):
        if self.session is None:
            from pose_retrieval import RetrievalSession

            self.session = RetrievalSession(self.output_dir)

        with tempfile.TemporaryDirectory(dir=self.workspace) as working:
            working = Path(working)
            source = working / f"upload{upload.suffix}"
            upload.rename(source)
            result = self.session.search(source, working, index=1, top_k=3, angle=angle)

            request_id = uuid.uuid4().hex
            assets = self.workspace / request_id
            assets.mkdir()

            def publish(path, name):
                save_preview(path, assets / f"{name}.jpg")
                return f"/uploads/{request_id}/{name}.jpg"

            try:
                matches = [{
                    "id": row["id"], "similarity": float(row["similarity"]),
                    **{key: row[key] for key in ("landmarks33", "projected_landmarks33",
                                                "visible_count", "estimated_count", "outside_count")},
                    "reference_url": publish(row["reference"], f"reference-{rank}"),
                    "overlay_url": publish(row["overlay"], f"overlay-{rank}"),
                } for rank, row in enumerate(result["matches"], 1)]
                excluded = []
                for rank, row in enumerate(result["excluded"], 1):
                    entry = {"id": row["id"], "reason": row["reason"]}
                    if rank <= 3 and row.get("reference"):
                        entry["reference_url"] = publish(row["reference"], f"excluded-{rank}")
                        if row.get("similarity") is not None:
                            entry["similarity"] = float(row["similarity"])
                    excluded.append(entry)
                response = {
                    "query_url": publish(result["query"], "query"),
                    "scene_matches": [{
                        "id": row["id"], "similarity": row["similarity"], "source": row["source"],
                        "reference_url": publish(row["reference"], f"scene-{rank}"),
                    } for rank, row in enumerate(result["scene_matches"], 1)],
                    "camera_label": result["camera_label"],
                    "framing_label": result["framing_label"],
                    "message": result["message"],
                    "excluded": excluded,
                    "matches": matches,
                }
            except Exception:
                shutil.rmtree(assets)
                raise

        self.recent_uploads.append(assets)
        if len(self.recent_uploads) > 8:
            shutil.rmtree(self.recent_uploads.popleft())
        return response


class RetrievalHandler(BaseHTTPRequestHandler):
    timeout = 15

    def send_content(self, status, content, mime):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def send_json(self, status, value):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_content(status, content, "application/json; charset=utf-8")

    def local_host(self):
        port = self.server.server_port
        return self.headers.get("Host", "").lower() in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.local_host():
            return self.send_error(403)

        path = unquote(urlsplit(self.path).path)
        if path in {"/pose_upload.js", "/pose_upload.css"}:
            file = TOOLS / path[1:]
        elif ASSET_ROUTE.fullmatch(path):
            file = self.server.workspace / path.removeprefix("/uploads/")
        else:
            if path.endswith("/"):
                path += "index.html"
            root = self.server.report_dir
            try:
                file = (root / path.lstrip("/")).resolve()
            except (OSError, RuntimeError, ValueError):
                return self.send_error(404)
            if not file.is_relative_to(root) or file.suffix not in REPORT_EXTENSIONS:
                return self.send_error(404)

        try:
            content = file.read_bytes()
        except (OSError, ValueError):
            return self.send_error(404)
        if file.suffix == ".json":
            return self.send_json(200, omit_file_paths(json.loads(content)))
        if b'id="pose-upload"' not in content:
            if path == "/index.html":
                includes = (b'<link rel="stylesheet" href="/pose_upload.css">'
                            b'<script type="module" src="/pose_upload.js"></script>')
                content = content.replace(b"</head>", includes + b"</head>", 1)
                content = content.replace(b"</header>", b'</header><section id="pose-upload"></section>', 1)
            elif path == "/empty-field/index.html":
                link = '<p><a href="/#pose-upload">用自己的照片找姿势</a></p>'.encode()
                content = content.replace(b"</header>", link + b"</header>", 1)
        self.send_content(200, content, mimetypes.guess_type(file)[0] or "application/octet-stream")

    do_HEAD = do_GET

    def do_POST(self):
        if not self.local_host():
            return self.send_json(403, {"error": "请从本机检索页面上传照片。"})
        url = urlsplit(self.path)
        if url.path != "/api/search":
            return self.send_json(404, {"error": "检索地址不存在。"})
        angle = parse_qs(url.query, keep_blank_values=True).get("angle", ["auto"])
        if len(angle) != 1 or angle[0] not in ANGLES:
            return self.send_json(400, {"error": "请选择有效的拍摄机位。"})
        if self.headers.get_content_type() not in UPLOAD_TYPES:
            return self.send_json(415, {"error": "请选择 JPEG、PNG 或 WebP 照片。"})
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
            return self.send_json(411, {"error": "无法确认上传的照片大小。"})
        try:
            size = int(lengths[0])
        except ValueError:
            size = 0
        if size <= 0:
            return self.send_json(400, {"error": "请选择一张照片。"})
        if size > MAX_UPLOAD:
            return self.send_json(413, {"error": "上传超过 64 MiB。请刷新页面，让浏览器自动缩小照片后重试。"})

        lock = self.server.model_lock
        if not lock.acquire(blocking=False):
            return self.send_json(503, {"error": "正在检索另一张照片，请稍后再试。"})
        try:
            status, response = self.run_search(size, angle[0])
        finally:
            lock.release()
        self.send_json(status, response)

    def run_search(self, size, angle):
        with tempfile.TemporaryDirectory(dir=self.server.workspace) as upload_dir:
            upload = Path(upload_dir) / "upload"
            try:
                remaining = size
                with upload.open("wb") as output:
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            return 400, {"error": "照片上传不完整，请重试。"}
                        output.write(chunk)
                        remaining -= len(chunk)
                with Image.open(upload, formats=tuple(FORMATS)) as image:
                    if image.width * image.height > MAX_PIXELS:
                        return 413, {"error": "原图超过 8000 万像素。请刷新页面后使用自动缩图。"}
                    if getattr(image, "is_animated", False):
                        return 400, {"error": "请选择一张静态照片。"}
                    if max(image.size) > 2560 or size > 8 * 1024 * 1024:
                        image.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                        source = upload.with_suffix(".jpg")
                        ImageOps.exif_transpose(image).convert("RGB").save(source, quality=90)
                    else:
                        image.load()
                        source = upload.rename(upload.with_suffix(FORMATS[image.format]))
            except Image.DecompressionBombError:
                return 413, {"error": "原图超过 8000 万像素。请刷新页面后使用自动缩图。"}
            except TimeoutError:
                return 408, {"error": "上传超时，请重试。"}
            except (OSError, SyntaxError, ValueError):
                return 400, {"error": "无法读取照片，请选择有效的 JPEG、PNG 或 WebP。"}

            try:
                return 200, self.server.search(source, angle)
            except Exception as error:
                self.log_error("Search failed: %s", error)
                return 500, {"error": "检索失败，请稍后重试。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4186)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not (output_dir / "report/index.html").is_file():
        parser.error(f"Report missing under {output_dir / 'report'}")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with tempfile.TemporaryDirectory(prefix="pose-retrieval-") as workspace:
        with RetrievalServer((args.host, args.port), output_dir, Path(workspace)) as server:
            print(f"Pose retrieval report ready at http://{args.host}:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
