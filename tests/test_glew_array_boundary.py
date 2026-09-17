"""I test production GL array conversion without loading a graphics driver."""
import os
import json
import sys
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GlewArrays(unittest.TestCase):
    def test_uploads(self):
        flags = subprocess.run(["pkg-config", "--cflags", "glew"], capture_output=True,
                               text=True, timeout=15)
        if flags.returncode:
            self.skipTest("GLEW SDK headers unavailable")
        with tempfile.TemporaryDirectory(prefix="nano-glew-arrays-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Wall", "-Wextra", "-Werror", "-Isrc", "-Isrc/runtime",
                *shlex.split(flags.stdout), "tests/test_glew_arrays.c",
                "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c",
                "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            metadata = json.loads((ROOT / "modules/glew/module.json").read_text())
            libraries = subprocess.run(["pkg-config", "--libs", "glew"], capture_output=True,
                                       text=True, timeout=15)
            self.assertEqual(libraries.returncode, 0, libraries.stderr)
            shared = (["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin"
                      else ["-shared", "-fPIC"])
            platform = ["-framework", "OpenGL"] if sys.platform == "darwin" else []
            artifact = Path(tmp) / "glew.so"
            linked = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", *shared,
                "-Isrc", "-Isrc/runtime", *shlex.split(flags.stdout),
                *[str(ROOT / "modules/glew" / name) for name in metadata["c_sources"]],
                *shlex.split(libraries.stdout), *platform, "-o", str(artifact)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            symbols = subprocess.run(["nm", "-g", str(artifact)], capture_output=True,
                                     text=True, timeout=15)
            self.assertEqual(symbols.returncode, 0, symbols.stderr)
            for name in ("nl_gl3_buffer_data_f32", "nl_gl3_buffer_data_u32"):
                self.assertIn(name + "__nano_array_abi", symbols.stdout)


if __name__ == "__main__":
    unittest.main()
