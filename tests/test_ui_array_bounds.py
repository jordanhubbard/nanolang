"""I exercise array guards with fake drawing calls, without opening a window."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UiArrayBounds(unittest.TestCase):
    def test_bounds(self):
        flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "sdl2", "SDL2_ttf"], text=True)
        with tempfile.TemporaryDirectory(prefix="nano-ui-arrays-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "tests/test_ui_array_bounds.c", "-o", str(output),
                *shlex.split(flags)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            env = dict(os.environ)
            if os.uname().sysname == "Darwin":
                libdir = subprocess.check_output(
                    ["pkg-config", "--variable=libdir", "sdl3"], text=True).strip()
                env["DYLD_LIBRARY_PATH"] = libdir + ":" + env.get("DYLD_LIBRARY_PATH", "")
            ran = subprocess.run([str(output)], env=env, capture_output=True,
                                 text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
