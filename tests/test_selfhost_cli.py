"""I check CLI rejection and C-source output before trusting driver success."""
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()
SOURCE = '''fn main() -> int { (println "cli-source-ok") return 0 }
shadow main { assert (== (main) 0) }
'''


class SelfhostCliTests(unittest.TestCase):
    def invoke(self, args, directory, cwd=ROOT):
        return subprocess.run([str(COMPILER), *map(str, args)], cwd=cwd,
                              env=dict(os.environ, TMPDIR=str(directory), NANO_CC="/bin/false"),
                              capture_output=True, text=True, timeout=60)

    def test_invalid_arguments_preserve_outputs(self):
        cases = [
            (["--unknown"], "I do not recognize"),
            (["--target", "unknown"], "I do not support"),
            (["--target", "riscv"], "I do not support"),
            (["--target"], "I need a value"),
            (["-o"], "I need a value"),
            (["--llm-diags-json"], "I need a value"),
            (["-o", "--help"], "I need a value"),
            (["--target", ""], "I need a value"),
            (["second.nano"], "I accept one input"),
        ]
        for args, diagnostic in cases:
            with self.subTest(args=args), tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
                directory = Path(tmp)
                source = directory / "input.nano"
                source.write_text(SOURCE)
                output = directory / "preserved"
                output.write_text("preserve me")
                result = self.invoke([source, "-o", output, *args], directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stdout + result.stderr)
                self.assertEqual(output.read_text(), "preserve me")
                self.assertEqual(source.read_text(), SOURCE)
                self.assertEqual(sorted(p.name for p in directory.iterdir()), ["input.nano", "preserved"])

    def test_source_output_without_native_compiler(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            output = directory / "output source.c"
            result = self.invoke([source, "--target", "c", "-o", output], directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            code = output.read_text()
            self.assertIn("int main(", code)
            self.assertIn("cli-source-ok", code)
            self.assertEqual(source.read_text(), SOURCE)
            binary = directory / "program"
            compiled = subprocess.run(["cc", "-O2", "-std=gnu11", "-I", str(ROOT / "src"),
                                       "-I", str(ROOT / "modules/std"), str(output), "-lm", "-o", str(binary)],
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(executed.stdout, "cli-source-ok\n")

    def test_default_source_path_and_option_order(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            result = self.invoke(["--target", "c", "--", source], directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("int main(", source.with_suffix(".c").read_text())

    def test_input_cannot_be_output(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            result = self.invoke([source, "--target", "c", "-o", source], directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I will not overwrite", result.stdout + result.stderr)
            self.assertEqual(source.read_text(), SOURCE)

    def test_help(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            result = self.invoke(["--help"], Path(tmp))
            self.assertEqual(result.returncode, 0)
            self.assertIn("--target", result.stdout)

    def test_missing_input(self):
        for args in ([], ["--target", "c"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
                directory = Path(tmp)
                result = self.invoke(args, directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No input file", result.stdout + result.stderr)
                self.assertEqual(list(directory.iterdir()), [])

    def test_option_delimiter_allows_hyphen_input(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "-input.nano"
            source.write_text(SOURCE)
            result = self.invoke(["--target", "c", "--", "-input.nano"], directory, cwd=directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("int main(", source.with_suffix(".c").read_text())

    def test_output_write_failure(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            result = self.invoke([source, "--target", "c", "-o", directory / "missing/out.c"], directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I could not write", result.stdout + result.stderr)

    def test_explicit_native_target(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-cli-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            binary = directory / "program"
            result = subprocess.run([str(COMPILER), "--target", "native", str(source), "-o", str(binary)],
                                    cwd=ROOT, env=dict(os.environ, TMPDIR=tmp, NANO_CC=shutil.which("cc")),
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(executed.stdout, "cli-source-ok\n")


if __name__ == "__main__":
    unittest.main()
