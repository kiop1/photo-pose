# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Estimate the apparent camera angle from a photo via local ShutterMuse."""
import argparse
import base64
from http.client import HTTPConnection, HTTPException
from io import BytesIO
import json
import os

from PIL import Image, ImageOps


SOURCE = "shuttermuse"
ANGLES = ("high", "level", "low", "unknown")
PROMPT = (
    "判断照片中的镜头视角是向下俯拍（high）、平拍（level）、向上仰拍（low），还是证据不足（unknown）。"
    "近看主体、地面和周围物体的透视：有无头顶或其他物体的上表面与向下看的地面，或下巴下侧或其他底面与向上的透视？"
    "平拍需要正面透视线索。判断相机对着主体或现场的上下角度，构图、画面旋转、人物视线或姿势本身不算证据。"
    "先用一句话写照片中实际可见的透视证据 evidence，再写 angle 和 confidence（0到1）。"
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "camera_view",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "evidence": {"type": "string"},
                "angle": {"type": "string", "enum": list(ANGLES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["evidence", "angle", "confidence"],
            "additionalProperties": False,
        },
    },
}


def _failure(error, evidence):
    return {"angle": "unknown", "confidence": 0.0, "source": SOURCE,
            "evidence": evidence, "error": error}


def estimate_camera_view(path, *, timeout=60.0):
    """Return angle, confidence, source and evidence. Failures include an error code."""
    try:
        with Image.open(path) as original:
            photo = ImageOps.exif_transpose(original).convert("RGB")
            photo.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            photo.save(buffer, format="JPEG", quality=88)
    except (OSError, ValueError, Image.DecompressionBombError):
        return _failure("image", "Could not decode the input as a photograph.")

    image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = json.dumps({
        "model": os.environ.get("SHUTTERMUSE_MODEL", "").strip() or SOURCE,
        "temperature": 0,
        "max_tokens": 160,
        "response_format": RESPONSE_FORMAT,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": PROMPT},
        ]}],
    })
    connection = HTTPConnection("127.0.0.1", 8081, timeout=timeout)
    try:
        connection.request("POST", "/v1/chat/completions", payload,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return _failure("http", f"Local ShutterMuse returned HTTP {response.status}.")
        raw = response.read(65536)
    except TimeoutError:
        return _failure("timeout", "Local ShutterMuse did not respond before the timeout.")
    except (OSError, HTTPException):
        return _failure("connection", "Could not reach local ShutterMuse on port 8081.")
    finally:
        connection.close()

    try:
        choice = json.loads(raw)["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Incomplete response")
        result = json.loads(choice["message"]["content"])
        angle, confidence, evidence = result["angle"], result["confidence"], result["evidence"]
        if (angle not in ANGLES or type(confidence) not in (float, int)
                or not 0 <= confidence <= 1 or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Invalid angle, confidence, or evidence")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return _failure("response", "Local ShutterMuse did not return a valid camera-view assessment.")

    # shortcut: self-reported confidence is uncalibrated, revisit with a labeled validation set.
    if confidence < 0.70:
        angle = "unknown"
    return {"angle": angle, "confidence": float(confidence), "source": SOURCE,
            "evidence": evidence.strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    for image in args.images:
        print(json.dumps({"image": image, **estimate_camera_view(image, timeout=args.timeout)},
                         ensure_ascii=False), flush=True)
