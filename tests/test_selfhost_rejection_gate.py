"""I distinguish semantic compiler refusals from infrastructure failures."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
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

    def test_shell_caller_preserves_prior_and_new_rejected_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="nano-selfhost-caller-") as directory:
            work = Path(directory)
            selfhost = work / "tests/selfhost"
            binary = work / "bin"
            selfhost.mkdir(parents=True)
            binary.mkdir()
            shutil.copy2(ROOT / "tests/selfhost/run_selfhost_tests.sh",
                         selfhost / "run_selfhost_tests.sh")
            shutil.copy2(GATE, selfhost / "expect_rejection.py")
            for suite in ("test_selfhost_import_paths.py", "test_selfhost_cli.py"):
                (work / "tests" / suite).write_text("raise SystemExit(0)\n")

            compiler = binary / "fake_nanoc"
            compiler.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                from pathlib import Path
                import sys

                source = Path(sys.argv[1]).name
                output = Path(sys.argv[sys.argv.index("-o") + 1])
                diagnostics = {
                    "test_requires_bool.nano": "[E0001] assert condition must be bool",
                    "test_function_arg_type_errors.nano": "[E0010] Argument 1 to 'add': expected int, got string",
                    "test_returned_function_arg_type_error.nano": "[E0010] Argument 1 to the function expression: expected int, got string",
                    "test_returned_function_arity_error.nano": "[E0010] The function expression expects 1 argument(s), but I see 2.",
                    "test_opaque_nonzero_argument.nano": "[E0010] Argument 1 to 'is_null': expected SDL_Window, got int",
                }
                if source in diagnostics:
                    if source == "test_requires_bool.nano":
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text("new rejected artifact")
                    print(diagnostics[source])
                    raise SystemExit(1)
                program = "#!/bin/sh\\n"
                if source == "test_returned_function_calls.nano":
                    program += "printf 'callee\\nargument\\n'\\n"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(program)
                output.chmod(0o755)
            """))
            compiler.chmod(0o755)

            prior = binary / "selfhost_test_requires_bool"
            prior.write_text("prior rejected artifact")
            result = subprocess.run(
                ["/bin/sh", "tests/selfhost/run_selfhost_tests.sh"],
                cwd=work,
                env={**os.environ, "NANOLANG_SELFHOST_COMPILER": str(compiler)},
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not the required semantic rejection", result.stdout)
            self.assertEqual(prior.read_text(), "prior rejected artifact")
            runs = list((work / ".test_output/selfhost").glob("negative.*"))
            self.assertEqual(len(runs), 1)
            rejected = runs[0] / "test_requires_bool/program"
            self.assertEqual(rejected.read_text(), "new rejected artifact")
            self.assertIn("assert condition must be bool",
                          (runs[0] / "test_requires_bool/compile.log").read_text())


if __name__ == "__main__":
    unittest.main()
