"""I distinguish semantic compiler refusals from infrastructure failures."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tests/selfhost/expect_rejection.py"


class SelfhostRejectionGateTests(unittest.TestCase):
    def run_gate(self, code: str, *, expected="semantic refusal", artifact=False,
                 timeout=2.0, executable=None):
        with tempfile.TemporaryDirectory(prefix="nano-selfhost-rejection-") as directory:
            work = Path(directory)
            log = work / "compiler.log"
            output = work / "program"
            if artifact:
                output.write_text("prior artifact")
            command = executable or [sys.executable, "-c", code]
            result = subprocess.run(
                [sys.executable, str(GATE), "--timeout", str(timeout),
                 "--log", str(log), "--output", str(output),
                 "--require", expected, "--", *command],
                cwd=ROOT, capture_output=True, text=True, timeout=5,
            )
            return result, (log.read_bytes() if log.exists() else b""), output.exists(), (
                output.read_bytes() if output.exists() else None
            )

    def test_exact_semantic_rejection_passes_and_retains_diagnostic(self):
        result, log, exists, _ = self.run_gate(
            "import sys; print('semantic refusal'); raise SystemExit(1)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log, b"semantic refusal\n")
        self.assertFalse(exists)

    def test_success_is_not_a_rejection_and_keeps_published_artifact(self):
        code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('built')"
        with tempfile.TemporaryDirectory(prefix="nano-selfhost-success-") as directory:
            work = Path(directory)
            log, output = work / "compiler.log", work / "program"
            result = subprocess.run(
                [sys.executable, str(GATE), "--timeout", "2", "--log", str(log),
                 "--output", str(output), "--require", "semantic refusal", "--",
                 sys.executable, "-c", code, str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "built")

    def test_mismatched_native_failure_is_not_a_semantic_rejection(self):
        result, log, exists, _ = self.run_gate(
            "import sys; print('ld: missing library'); raise SystemExit(1)"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(log, b"ld: missing library\n")
        self.assertFalse(exists)

    def test_signal_is_not_a_rejection(self):
        result, _, exists, _ = self.run_gate(
            "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("died from signal", result.stderr)
        self.assertFalse(exists)

    def test_timeout_is_not_a_rejection_and_retains_partial_output(self):
        result, log, exists, _ = self.run_gate(
            "import time; print('semantic refusal', flush=True); time.sleep(2)",
            timeout=0.05,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeded 0.05 seconds", result.stderr)
        self.assertEqual(log, b"semantic refusal\n")
        self.assertFalse(exists)

    def test_launch_failure_is_not_a_rejection_and_is_logged(self):
        missing = "/definitely/absent/nanolang-compiler"
        result, log, exists, _ = self.run_gate("", executable=[missing])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not launch", result.stderr)
        self.assertIn(missing.encode(), log)
        self.assertFalse(exists)

    def test_rejection_that_publishes_an_artifact_fails_and_preserves_it(self):
        code = (
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('diagnostic artifact'); "
            "print('semantic refusal'); raise SystemExit(1)"
        )
        with tempfile.TemporaryDirectory(prefix="nano-selfhost-artifact-") as directory:
            work = Path(directory)
            log, output = work / "compiler.log", work / "program"
            result = subprocess.run(
                [sys.executable, str(GATE), "--timeout", "2", "--log", str(log),
                 "--output", str(output), "--require", "semantic refusal", "--",
                 sys.executable, "-c", code, str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "diagnostic artifact")
            self.assertEqual(log.read_bytes(), b"semantic refusal\n")

    def test_preexisting_artifact_is_refused_without_modification(self):
        result, log, exists, content = self.run_gate(
            "raise SystemExit(1)", artifact=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(log)
        self.assertTrue(exists)
        self.assertEqual(content, b"prior artifact")


if __name__ == "__main__":
    unittest.main()
