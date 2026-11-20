#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


HOST = "127.0.0.1"
PORT = "8081"
GGUF_URL = "https://huggingface.co/prithivMLmods/ShutterMuse-GGUF/resolve/df18a14407dd0f95e4266883649060cb3a63a4cd"
GGUF_FILES = (
    ("ShutterMuse.Q4_K_M.gguf", 5027785312, "539701f6def8e87a5a596b6220a8eae8848358c77eb36223e551a590cda200f4"),
    ("ShutterMuse.mmproj-bf16.gguf", 1162569248, "f4a68d229a96ff68ffb3efa98907f409cb17dbad8f9b99543c912c0e02c6c20b"),
)
MLX_SOURCE = "ShutterMuse/ShutterMuse"
MLX_REVISION = "cea60355d16454392b8b94e689bb8794e9172fb2"
MLX_PACKAGE = "mlx-vlm==0.8.0"


def require_command(name, install):
    command = shutil.which(name)
    if command is None:
        raise RuntimeError(f"需要 {name}。安装命令：{install}")
    return command


def file_matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest


def download_gguf(directory, name, size, digest):
    destination = directory / name
    if file_matches(destination, size, digest):
        return destination

    partial = directory / f"{name}.part"
    if not file_matches(partial, size, digest):
        if partial.exists() and partial.stat().st_size >= size:
            partial.unlink()
        curl = require_command("curl", "brew install curl")
        print(f"下载 {name} ({size / 1e9:.2f} GB)", flush=True)
        command = [
            curl, "--fail", "--location", "--show-error",
            "--progress-bar" if sys.stderr.isatty() else "--silent",
            "--connect-timeout", "20", "--retry", "2",
            "--output", str(partial), f"{GGUF_URL}/{name}",
        ]
        result = subprocess.run([*command, "--continue-at", "-"])
        if result.returncode == 33:
            # curl received a response that did not honor the resume range.
            subprocess.run(command, check=True)
        else:
            result.check_returncode()
        if not file_matches(partial, size, digest):
            if partial.exists() and partial.stat().st_size >= size:
                partial.unlink()
            raise RuntimeError(f"{name} 未通过完整性校验，请重新运行。")

    partial.replace(destination)
    return destination


def llama_command(directory):
    server = require_command("llama-server", "brew install llama.cpp")
    model, projector = (download_gguf(directory, *file) for file in GGUF_FILES)
    return [
        server, "--model", str(model), "--mmproj", str(projector),
        "--alias", "shuttermuse", "--ctx-size", "4096", "--parallel", "1",
        "--jinja", "--no-webui",
    ]


def mlx_command(directory, env):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX 后端需要 Apple 芯片和原生 arm64 环境。")
    uv = require_command("uv", "brew install uv")
    env.setdefault("UV_CACHE_DIR", str(directory / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(directory / "python"))
    env["HF_HUB_CACHE"] = str(directory / "hub")
    env["HF_XET_CACHE"] = str(directory / "xet")
    run = [uv, "run", "--no-project", "--python", "3.12", "--with", MLX_PACKAGE, "--with", "socksio"]
    model = directory / "ShutterMuse-4bit"
    if not model.is_dir():
        partial = directory / "ShutterMuse-4bit.partial"
        if partial.exists():
            shutil.rmtree(partial)
        print("将官方 ShutterMuse 转换为 MLX 4-bit。首次下载约 17.6 GB。", flush=True)
        subprocess.run([
            *run, "mlx_vlm.convert", "--hf-path", MLX_SOURCE, "--revision", MLX_REVISION,
            "--mlx-path", str(partial), "--quantize", "--q-bits", "4", "--q-group-size", "64",
        ], check=True, env=env)
        partial.rename(model)

    env["HF_HUB_CACHE"] = str(directory / "serve-hub")
    env["HF_HUB_OFFLINE"] = "1"
    # MLX-VLM's image-decoding errors can include the original data URI.
    return [*run, "mlx_vlm.server", "--model", str(model), "--log-level", "CRITICAL"]


def main():
    parser = argparse.ArgumentParser(description=f"在 {HOST}:{PORT} 启动 ShutterMuse 模型服务。")
    parser.add_argument("backend", choices=("llama", "mlx"))
    args = parser.parse_args()
    default_cache = Path(__file__).resolve().parents[1] / ".pose-models"
    cache = Path(os.environ.get("SHUTTERMUSE_CACHE_DIR") or default_cache).expanduser().resolve()
    directory = cache / args.backend
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)

    env = os.environ.copy()
    with (directory / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = llama_command(directory) if args.backend == "llama" else mlx_command(directory, env)
    command.extend(("--host", HOST, "--port", PORT))
    print(f"启动 {args.backend}，接口 http://{HOST}:{PORT}/v1", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"ShutterMuse 启动失败：{error}")
    except KeyboardInterrupt:
        sys.exit(130)
