"""I exercise the real runner's reporting, not compiler conformance."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests/cross-backend/run-all.sh"
PAYLOAD = '''import os, sys
print(os.environ.get("FAKE_OUTPUT", "ok"))
if os.environ.get("FAKE_STATUS", "0") != "0":
    print("fixture execution failed", file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_STATUS", "0")))
'''


class CrossBackendRunnerTests(unittest.TestCase):
    def run_case(self, backend="c", status=0, output="ok", xfail=False,
                 compile_failure=False):
        with tempfile.TemporaryDirectory(prefix="nanolang-runner-test-") as directory:
            root = Path(directory)
            suite = root / "tests/cross-backend"
            suite.mkdir(parents=True)
            runner = suite / "run-all.sh"
            shutil.copyfile(RUNNER, runner)
            (suite / "case.nano").write_text("# I am a controlled compiler fixture.\n")
            (suite / "case.expected").write_text("ok\n")
            if xfail:
                (suite / "case.xfail").write_text(backend + "\n")
            binaries = root / "tools"
            binaries.mkdir()
            for name in ("bash", "dirname", "basename", "grep", "cat", "mktemp", "mkdir", "rm"):
                (binaries / name).symlink_to(shutil.which(name))
            script = f"#!{sys.executable}\n" + f'''
import os, sys
from pathlib import Path
kind = Path(sys.argv[0]).name
if kind == "compiler":
    if os.environ.get("FAKE_COMPILE_FAILURE") == "1":
        print("fixture compile failed", file=sys.stderr)
        sys.exit(1)
    Path(sys.argv[sys.argv.index("-o") + 1]).write_text(".text\\n.target sm_70\\n")
elif kind == "gcc":
    target = Path(sys.argv[sys.argv.index("-o") + 1])
    target.write_text({("#!" + sys.executable + chr(10) + PAYLOAD)!r})
    target.chmod(0o700)
else:
    exec({PAYLOAD!r})
'''
            for name in ("compiler", "gcc"):
                tool = binaries / name
                tool.write_text(script)
                tool.chmod(0o700)
            scratch = root / "scratch space"
            scratch.mkdir()
            env = dict(os.environ, PATH=str(binaries), TMPDIR=str(scratch),
                       NANOLANG_TEST_BACKENDS=backend, FAKE_STATUS=str(status),
                       FAKE_OUTPUT=output, FAKE_COMPILE_FAILURE=str(int(compile_failure)))
            result = subprocess.run([str(binaries / "bash"), str(runner), str(binaries / "compiler")],
                                    env=env, cwd=root, capture_output=True, text=True, timeout=20)
            self.assertEqual(list(scratch.iterdir()), [], "I must clean private scratch files")
            return result

    def test_successful_execution(self):
        result = self.run_case()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS:          1", result.stdout)

    def test_matching_stdout_does_not_hide_failure(self):
        result = self.run_case(status=7)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL:          1", result.stdout)
        self.assertIn("fixture execution failed", result.stderr)

    def test_retired_backends_refuse_before_compilation(self):
        for backend in ("llvm", "wasm", "c wasm"):
            with self.subTest(backend=backend):
                result = self.run_case(backend, compile_failure=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("I retired direct AST", result.stderr)
                self.assertNotIn("fixture compile failed", result.stderr)
                self.assertNotIn("PASS:", result.stdout)

    def test_wrong_output_fails(self):
        result = self.run_case(output="wrong")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output mismatch", result.stdout)

    def test_compilation_failure(self):
        result = self.run_case(compile_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixture compile failed", result.stderr)

    def test_expected_failure_is_not_pass(self):
        result = self.run_case(status=7, xfail=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("XFAIL:         1", result.stdout)
        self.assertIn("PASS:          0", result.stdout)

    def test_unexpected_pass_fails(self):
        result = self.run_case(xfail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("XPASS:         1", result.stdout)

    def test_validation_is_not_execution(self):
        for backend in ("riscv", "ptx"):
            with self.subTest(backend=backend):
                result = self.run_case(backend)
                self.assertEqual(result.returncode, 0)
                self.assertIn("VALIDATE-ONLY: 1", result.stdout)
                self.assertIn("PASS:          0", result.stdout)

    def test_auxiliary_build_contract(self):
        help_result = subprocess.run(["make", "help"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=20)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("EXAMPLES_BACKEND=c|native|nanoisa|vm", help_result.stdout)
        self.assertNotIn("Build examples as LLVM IR", help_result.stdout)
        self.assertNotIn("Build examples as WASM", help_result.stdout)
        bundle = ROOT / "examples/playground/public"
        before = {p.name: p.read_bytes() for p in bundle.glob("nanolang.*")}
        result = subprocess.run(["make", "wasm-playground"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot rebuild the historical browser interpreter", result.stderr)
        self.assertIn("test-nvm2wasm", result.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in bundle.glob("nanolang.*")})

    def test_invalid_selection_fails(self):
        for backend in ("", "   ", "typo", "c typo"):
            with self.subTest(backend=backend):
                result = self.run_case(backend)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
