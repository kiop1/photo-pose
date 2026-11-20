import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[1] / "tools" / "shuttermuse-server.py"
spec = importlib.util.spec_from_file_location("shuttermuse_server", source)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()

    def test_launches_only_the_selected_backend_on_loopback(self):
        with patch.dict(launcher.os.environ, {"SHUTTERMUSE_CACHE_DIR": str(self.directory)}), patch.object(
            launcher.sys, "argv", ["shuttermuse-server.py", "llama"]
        ), patch.object(launcher, "llama_command", return_value=["llama-server"]) as llama, patch.object(
            launcher, "mlx_command"
        ) as mlx, patch.object(launcher.os, "execvpe") as execute:
            launcher.main()
            llama.assert_called_once_with(self.directory / "llama")
            mlx.assert_not_called()
            self.assertEqual(execute.call_args.args[1], ["llama-server", "--host", "127.0.0.1", "--port", "8081"])

    def test_resumes_validates_and_reuses_gguf(self):
        contents = b"GGUF test data"
        digest = hashlib.sha256(contents).hexdigest()
        partial = self.directory / "example.gguf.part"
        partial.write_bytes(contents[:4])

        def download(command):
            self.assertIn("--continue-at", command)
            with partial.open("ab") as file:
                file.write(contents[4:])
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launcher, "require_command", return_value="curl"), patch.object(
            launcher.subprocess, "run", side_effect=download
        ) as curl:
            for _ in range(2):
                cached = launcher.download_gguf(self.directory, "example.gguf", len(contents), digest)
                self.assertEqual(cached.read_bytes(), contents)
            curl.assert_called_once()
        self.assertFalse(partial.exists())

    def test_rejects_bad_hash(self):
        def download(command):
            partial.write_bytes(b"bad")
            return subprocess.CompletedProcess(command, 0)

        partial = self.directory / "example.gguf.part"
        digest = hashlib.sha256(b"GGU").hexdigest()
        with patch.object(launcher, "require_command", return_value="curl"), patch.object(
            launcher.subprocess, "run", side_effect=download
        ), self.assertRaises(RuntimeError):
            launcher.download_gguf(self.directory, "example.gguf", 3, digest)
        self.assertFalse(partial.exists())
        self.assertFalse((self.directory / "example.gguf").exists())

    def test_mlx_restarts_interrupted_conversion_then_reuses_model(self):
        partial = self.directory / "ShutterMuse-4bit.partial"
        partial.mkdir()
        env = {}

        def convert(command, **kwargs):
            self.assertFalse(partial.exists())
            self.assertIn(("--with", "socksio"), zip(command, command[1:]))
            self.assertIn("ShutterMuse/ShutterMuse", command)
            self.assertEqual(command[command.index("--q-bits") + 1], "4")
            partial.mkdir()

        with patch.object(launcher.platform, "system", return_value="Darwin"), patch.object(
            launcher.platform, "machine", return_value="arm64"
        ), patch.object(launcher, "require_command", return_value="uv"), patch.object(
            launcher.subprocess, "run", side_effect=convert
        ) as conversion:
            for _ in range(2):
                command = launcher.mlx_command(self.directory, env.copy())
                self.assertIn("mlx_vlm.server", command)
                self.assertEqual(command[command.index("--model") + 1], str(self.directory / "ShutterMuse-4bit"))
            conversion.assert_called_once()
        self.assertFalse(partial.exists())
