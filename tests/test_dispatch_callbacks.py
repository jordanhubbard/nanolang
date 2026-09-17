"""I check real dispatch captures, shadow execution, and isolated-call refusal."""
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(platform.system() == "Darwin", "I require Apple libdispatch")
class DispatchCallbacks(unittest.TestCase):
    def test_captures_and_isolation(self):
        with tempfile.TemporaryDirectory(prefix="nano-dispatch-") as directory:
            output = Path(directory) / "callbacks.nvm"
            compiled = subprocess.run([
                str(ROOT / "bin/nano_virt"), "tests/nanovm/dispatch_callbacks.nano",
                "--emit-nvm", "-o", str(output)], cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)],
                                      cwd=ROOT, capture_output=True, timeout=20)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            isolated = subprocess.run([str(ROOT / "bin/nano_vm"), "--isolate-ffi", str(output)],
                                      cwd=ROOT, capture_output=True, timeout=20)
            self.assertNotEqual(isolated.returncode, 0)
            self.assertIn(b"isolated FFI", isolated.stderr)


if __name__ == "__main__":
    unittest.main()
