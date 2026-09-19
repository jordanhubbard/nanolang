"""I retain ordered edit events beside generic SDL event consumers."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SdlTextInputEvents(unittest.TestCase):
    def test_mirrored_edit_stream(self):
        flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "sdl2"], text=True)
        with tempfile.TemporaryDirectory(prefix="nano-sdl-text-events-") as tmp:
            output = Path(tmp) / "probe"
            built = subprocess.run([
                *shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                "-Isrc", "tests/test_sdl_text_input_events.c",
                "src/runtime/dyn_array.c", "src/runtime/gc.c",
                "src/runtime/gc_struct.c", "src/utf8.c", "-o", str(output),
                *shlex.split(flags), "-lm"], cwd=ROOT, capture_output=True,
                text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(output)], cwd=ROOT,
                                 env=os.environ.copy(),
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == "__main__":
    unittest.main()
