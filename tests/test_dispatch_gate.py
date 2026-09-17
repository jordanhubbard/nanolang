"""I ensure compile failures and stalled VMs cannot produce a green comparison."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name("dispatch_equivalence.py")


class DispatchGate(unittest.TestCase):
    def check_case(self, mode, expected, sources=True):
        with tempfile.TemporaryDirectory(prefix="nano-dispatch-gate-") as tmp:
            root = Path(tmp)
            compiler = root / "compiler"
            compiler.write_text("#!" + sys.executable + "\n"
                "import pathlib,sys,time\n"
                f"mode={mode!r}\n"
                "if mode=='compile_timeout': time.sleep(5)\n"
                "if mode=='compile_fail': print('compile diagnostic'); sys.exit(7)\n"
                "if mode!='missing': pathlib.Path(sys.argv[3]).write_text('fixture')\n")
            for name in ("goto", "switch"):
                vm = root / name
                vm.write_text("#!" + sys.executable + "\n"
                    "import os,signal,sys,time\n"
                    f"mode={mode!r}\n"
                    "if mode=='vm_timeout': time.sleep(5)\n"
                    "if mode=='signal': os.kill(os.getpid(), signal.SIGTERM)\n"
                    "print('output', end='\\n\\n' if mode=='newline' and sys.argv[0].endswith('switch') else '\\n')\n"
                    "sys.exit(3 if mode=='status' and sys.argv[0].endswith('switch') else 0)\n")
                vm.chmod(0o700)
            compiler.chmod(0o700)
            source = root / "fixture.nano"
            source.touch()
            result = subprocess.run([sys.executable, str(RUNNER),
                "--compiler", str(compiler), "--vm-goto", str(root/"goto"),
                "--vm-switch", str(root/"switch"), "--logs", str(root/"logs"),
                "--timeout", "0.3", *([str(source)] if sources else [])],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            if sources:
                self.assertIn("0 skipped", result.stdout)
                self.assertTrue(list((root/"logs").glob("*.compile.log")))

    def test_success(self):
        self.check_case("ok", 0)

    def test_failures(self):
        for mode in ("compile_fail", "missing", "compile_timeout",
                     "vm_timeout", "signal", "newline", "status"):
            with self.subTest(mode=mode):
                self.check_case(mode, 1)

    def test_empty(self):
        self.check_case("ok", 1, sources=False)


if __name__ == "__main__":
    unittest.main()
