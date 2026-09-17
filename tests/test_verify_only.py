"""I verify modules without executing code or loading foreign libraries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VM = ROOT / "bin/nano_vm"


class VerifyOnly(unittest.TestCase):
    def compile(self, directory, source):
        path = directory / "input.nano"
        path.write_text(source)
        output = directory / "input.nvm"
        result = subprocess.run([str(ROOT / "bin/nano_virt"), str(path),
                                 "--emit-nvm", "-o", str(output)], cwd=ROOT,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output

    def verify(self, path, *options):
        return subprocess.run([str(VM), "--verify-only", *options, str(path)],
                              cwd=ROOT, capture_output=True, timeout=10)

    def test_does_not_execute(self):
        with tempfile.TemporaryDirectory(prefix="nano-verify-only-") as tmp:
            path = self.compile(Path(tmp),
                'fn main() -> int { (println "I executed") return 23 }')
            verified = self.verify(path)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(verified.stdout, b"")
            self.assertEqual(verified.stderr, b"")
            run = subprocess.run([str(VM), str(path)], cwd=ROOT,
                                 capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 23, run.stderr)
            self.assertIn(b"I executed", run.stdout)

    def test_unresolved_foreign_function_is_not_loaded(self):
        with tempfile.TemporaryDirectory(prefix="nano-verify-foreign-") as tmp:
            path = self.compile(Path(tmp),
                'extern fn missing_verify_only_symbol() -> int\n'
                'fn main() -> int { unsafe { return (missing_verify_only_symbol) } }')
            result = self.verify(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout + result.stderr, b"")

    def test_invalid_file_and_conflicting_modes(self):
        with tempfile.TemporaryDirectory(prefix="nano-verify-invalid-") as tmp:
            directory = Path(tmp)
            self.assertNotEqual(self.verify(directory / "missing.nvm").returncode, 0)
            bad = directory / "bad.nvm"
            bad.write_bytes(b"not a module")
            self.assertNotEqual(self.verify(bad).returncode, 0)
            path = self.compile(directory, 'fn main() -> int { return 0 }')
            for options in [("--daemon",), ("--cop",), ("--repeat", "2"),
                            ("--profile-isa", str(directory / "profile.json"))]:
                result = self.verify(path, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"execution options", result.stderr)
            self.assertFalse((directory / "profile.json").exists())


if __name__ == "__main__":
    unittest.main()
