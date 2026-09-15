"""I exercise retained post-mix callbacks with a real SDL dummy audio device."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MixerCallbacks(unittest.TestCase):
    def test_audio_callbacks_and_isolation(self):
        env = dict(os.environ, SDL_AUDIODRIVER="dummy")
        with tempfile.TemporaryDirectory(prefix="nano-mixer-") as directory:
            output = Path(directory) / "callbacks.nvm"
            compiled = subprocess.run([
                str(ROOT / "bin/nano_virt"), "tests/nanovm/mixer_callbacks.nano",
                "--emit-nvm", "-o", str(output)], cwd=ROOT, env=env,
                capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)],
                                      cwd=ROOT, env=env, capture_output=True, timeout=20)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            isolated = subprocess.run([str(ROOT / "bin/nano_vm"), "--isolate-ffi", str(output)],
                                      cwd=ROOT, env=env, capture_output=True, timeout=20)
            self.assertNotEqual(isolated.returncode, 0)
            self.assertIn(b"isolated FFI", isolated.stderr)


if __name__ == "__main__":
    unittest.main()
