"""I test host buffers without requiring a GPU or loading SDL's constructor."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArrayAdapterBoundaries(unittest.TestCase):
    def compile_run(self, source, flags=(), include=None):
        with tempfile.TemporaryDirectory(prefix="nano-array-adapter-") as tmp:
            directory = Path(tmp)
            if include is not None:
                (directory / "sdl_update_under_test.c").write_text(include)
            output = directory / "probe"
            command = [*shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                       "-Wall", "-Wextra", "-Werror", "-Isrc", "-I" + tmp,
                       source, *flags, "-o", str(output)]
            built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
            ran = subprocess.run([output], capture_output=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr.decode(errors="replace"))

    def test_gpu_drivers(self):
        links = ["-ldl"] if sys.platform.startswith("linux") else []
        for unified in (False, True):
            with self.subTest(unified=unified):
                self.compile_run("tests/test_gpu_array_boundary.c",
                                 [*links, *(["-DTEST_UNIFIED_GPU"] if unified else [])])

    def test_sdl_upload(self):
        flags = subprocess.run(["pkg-config", "--cflags", "sdl2"], capture_output=True, text=True, timeout=15)
        if flags.returncode:
            self.skipTest("SDL2 headers are unavailable")
        production = (ROOT / "modules/sdl_helpers/sdl_helpers.c").read_text()
        start = production.index("int64_t nl_sdl_update_texture(")
        end = production.index("int64_t nl_sdl_render_texture(", start)
        self.compile_run("tests/test_sdl_array_boundary.c", shlex.split(flags.stdout), production[start:end])
        checked = subprocess.run([*shlex.split(os.environ.get("CC", "cc")), "-std=c99",
                                  "-fsyntax-only", *shlex.split(flags.stdout),
                                  "modules/sdl_helpers/sdl_helpers.c"], cwd=ROOT, capture_output=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
