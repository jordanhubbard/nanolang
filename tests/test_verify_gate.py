"""I reject missing coverage and failures without depending on diagnostic wording."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import unittest

from dispatch_equivalence import execute

RUNNER = Path(__file__).with_name("verify_corpus.py")


class VerifyGate(unittest.TestCase):
    def check_case(self, mode, expected, sources=True):
        with tempfile.TemporaryDirectory(prefix="nano-verifier-gate-") as tmp:
            root = Path(tmp)
            compiler = root / "compiler"
            compiler.write_text("#!" + sys.executable + "\n"
                "import pathlib,sys,time\n"
                f"mode={mode!r}\n"
                "if mode=='compile_timeout': time.sleep(10)\n"
                "if mode=='compile_fail': sys.exit(7)\n"
                "if mode!='missing': pathlib.Path(sys.argv[3]).write_text('fixture')\n")
            vm = root / "vm"
            vm.write_text("#!" + sys.executable + "\n"
                "import os,signal,sys,time\n"
                "assert sys.argv[1]=='--verify-only'\n"
                f"mode={mode!r}\n"
                "if mode=='verify_timeout': time.sleep(10)\n"
                "if mode=='signal': os.kill(os.getpid(), signal.SIGTERM)\n"
                "print('unrelated diagnostic')\n"
                "sys.exit(4 if mode=='verify_fail' else 0)\n")
            compiler.chmod(0o700)
            vm.chmod(0o700)
            source = root / "fixture.nano"
            source.touch()
            logs = root / "logs"
            logs.mkdir()
            (logs / "0000-fixture.nvm").write_text("stale artifact")
            result = subprocess.run([sys.executable, str(RUNNER),
                "--compiler", str(compiler), "--vm", str(vm),
                "--logs", str(logs), "--timeout", "1",
                *([str(source)] if sources else [])],
                capture_output=True, text=True, timeout=8)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            if sources:
                self.assertIn("1 selected", result.stdout)
                self.assertIn("0 skipped", result.stdout)
                self.assertTrue((logs / "0000-fixture.compile.log").is_file())
                if mode in ("compile_fail", "missing", "compile_timeout"):
                    self.assertFalse((logs / "0000-fixture.verify.log").exists())
                else:
                    self.assertTrue((logs / "0000-fixture.verify.log").is_file())

    def test_success(self):
        self.check_case("ok", 0)

    def test_failures(self):
        for mode in ("compile_fail", "missing", "compile_timeout",
                     "verify_timeout", "verify_fail", "signal"):
            with self.subTest(mode=mode):
                self.check_case(mode, 1)

    def test_empty(self):
        self.check_case("ok", 1, sources=False)

    def test_detached_descendant_cannot_hold_capture_open(self):
        with tempfile.TemporaryDirectory(prefix="nano-verify-descendant-") as tmp:
            root = Path(tmp)
            pid_file = root / "child.pid"
            script = ("import pathlib,subprocess,sys\n"
                      "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)\n"
                      "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
                      "print('done')\n")
            try:
                status, output = execute([sys.executable, "-c", script, str(pid_file)],
                                         2, root / "output.log")
                self.assertEqual(status, 0)
                self.assertEqual(output, b"done\n")
            finally:
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
