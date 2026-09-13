"""I check CLI rejection and C-source output before trusting driver success."""
import os
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()
SOURCE = '''fn main() -> int { (println "cli-source-ok") return 0 }
shadow main { assert (== (main) 0) }
'''


class SelfhostCliTests(unittest.TestCase):
    def test_destination_probe_lifecycle_and_failures(self):
        with tempfile.TemporaryDirectory(prefix="nano-destination-unit-") as tmp:
            directory = Path(tmp)
            executable = directory / "probe-test"
            fixture = directory / "fixture"
            fixture.mkdir()
            stripping = "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
            command = ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                       "-ffunction-sections", "-fdata-sections", stripping,
                       "-I", str(ROOT / "src"),
                       str(ROOT / "tests/test_fs_destination_identity.c"), "-o", str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(executable), str(fixture)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(fixture.iterdir()), [])

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

    def test_artifact_report_collisions(self):
        for target in ("c", "native"):
            for kind in ("equal", "relative", "symlink", "hardlink", "missing", "parent_alias"):
                with self.subTest(target=target, kind=kind), tempfile.TemporaryDirectory(prefix="nano-output-alias-") as tmp:
                    directory = Path(tmp)
                    source = directory / "input.nano"
                    source.write_text(SOURCE)
                    output = directory / "output"
                    existing = kind not in ("missing", "parent_alias")
                    if existing:
                        output.write_bytes(b"prior artifact")
                    report = output
                    if kind == "relative":
                        report = Path("./output")
                    elif kind in ("symlink", "hardlink"):
                        report = directory / "report"
                        if kind == "symlink":
                            report.symlink_to(output)
                        else:
                            os.link(output, report)
                    elif kind == "parent_alias":
                        parent = directory / "alias"
                        parent.symlink_to(directory, target_is_directory=True)
                        report = parent / "output"
                    result = self.invoke([source, "--target", target, "-o", output,
                                          "--llm-diags-json", report], directory, cwd=directory)
                    self.assertEqual(source.read_text(), SOURCE)
                    if existing:
                        self.assertEqual(output.read_bytes(), b"prior artifact")
                    else:
                        self.assertFalse(output.exists())
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("I require separate artifact and diagnostic destinations", result.stdout + result.stderr)
                    self.assertFalse(list(directory.glob("nano_native_*")))

    def test_distinct_artifact_and_report(self):
        for target in ("c", "native"):
            for existing in (False, True):
                with self.subTest(target=target, existing=existing), tempfile.TemporaryDirectory(prefix="nano-output-alias-") as tmp:
                    directory = Path(tmp)
                    source = directory / "input.nano"
                    source.write_text(SOURCE)
                    output = directory / "output"
                    report = directory / "report.json"
                    if existing:
                        output.write_text("prior artifact")
                        report.write_text("prior report")
                    result = subprocess.run([str(COMPILER), str(source), "--target", target,
                                             "-o", str(output), "--llm-diags-json", str(report)],
                                            cwd=ROOT, env=dict(os.environ, TMPDIR=tmp, NANO_CC=shutil.which("cc")),
                                            capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue(json.loads(report.read_text())["success"])
                    if target == "c":
                        self.assertIn("int main(", output.read_text())
                    else:
                        executed = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                        self.assertEqual(executed.returncode, 0, executed.stderr)
                        self.assertEqual(executed.stdout, "cli-source-ok\n")
                    self.assertEqual(source.read_text(), SOURCE)

    def test_missing_destination_filesystem_name_equivalence(self):
        pairs = (("Output", "output"), ("\u00e9", "e\u0301"))
        for first, second in pairs:
            for target in ("c", "native"):
                with self.subTest(first=first, second=second, target=target), tempfile.TemporaryDirectory(prefix="nano-name-equivalence-") as tmp:
                    directory = Path(tmp)
                    source = directory / "input.nano"
                    source.write_text(SOURCE)
                    output = directory / first
                    report = directory / second
                    # I measure this filesystem, not the host platform's default.
                    output.write_text("identity probe")
                    aliases = report.exists() and os.path.samefile(output, report)
                    output.unlink()
                    result = subprocess.run([str(COMPILER), str(source), "--target", target,
                                             "-o", str(output), "--llm-diags-json", str(report)],
                                            cwd=ROOT, env=dict(os.environ, TMPDIR=tmp, NANO_CC=shutil.which("cc")),
                                            capture_output=True, text=True, timeout=120)
                    if aliases:
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("I require separate", result.stdout + result.stderr)
                        self.assertFalse(output.exists())
                        self.assertFalse(report.exists())
                        self.assertEqual(list(directory.iterdir()), [source])
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertTrue(json.loads(report.read_text())["success"])
                        if target == "c":
                            self.assertIn("int main(", output.read_text())
                        else:
                            executed = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                            self.assertEqual(executed.returncode, 0, executed.stderr)
                            self.assertEqual(executed.stdout, "cli-source-ok\n")
                    self.assertEqual(source.read_text(), SOURCE)

    def test_output_collision_precedes_parse_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="nano-output-alias-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text("fn main( -> invalid")
            output = directory / "output"
            output.write_text("prior artifact")
            result = self.invoke([source, "--target", "c", "-o", output,
                                  "--llm-diags-json", output], directory)
            self.assertEqual(output.read_text(), "prior artifact")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I require separate", result.stdout + result.stderr)

    def test_dangling_report_is_rejected_without_creating_target(self):
        with tempfile.TemporaryDirectory(prefix="nano-output-alias-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            output = directory / "output"
            report = directory / "report"
            report.symlink_to(output.name)
            result = self.invoke([source, "--target", "c", "-o", output,
                                  "--llm-diags-json", report], directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I cannot check artifact/diagnostic", result.stdout + result.stderr)
            self.assertFalse(output.exists())
            self.assertTrue(report.is_symlink())

    def test_source_aliases_are_rejected_before_writes(self):
        for target in ("c", "native"):
            for alias_kind in ("relative", "symlink", "hardlink"):
                for dependency in (False, True):
                    for report in (False, True):
                        with self.subTest(target=target, alias=alias_kind, dependency=dependency, report=report), tempfile.TemporaryDirectory(prefix="nano-source-alias-") as tmp:
                            directory = Path(tmp)
                            source = directory / "input.nano"
                            leaf = directory / "leaf.nano"
                            leaf_text = 'pub fn answer() -> int { return 42 }\nshadow answer { assert true }\n'
                            leaf.write_text(leaf_text)
                            text = (f'module "{leaf}" as leaf\n' if dependency else '') + SOURCE
                            source.write_text(text)
                            protected = leaf if dependency else source
                            alias = directory / "alias.nano"
                            if alias_kind == "relative":
                                alias = Path(protected.name)
                            elif alias_kind == "symlink":
                                alias.symlink_to(protected)
                            else:
                                os.link(protected, alias)
                            output = directory / "output"
                            output.write_bytes(b"prior output")
                            args = [source, "--target", target, "-o", output if report else alias]
                            if report:
                                args.extend(["--llm-diags-json", alias])
                            result = self.invoke(args, directory, cwd=directory)
                            self.assertEqual(source.read_text(), text)
                            self.assertEqual(leaf.read_text(), leaf_text)
                            self.assertEqual(output.read_bytes(), b"prior output")
                            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertIn("I will not overwrite", result.stdout + result.stderr)
                            self.assertFalse(list(directory.glob("nano_native_*")))

    def test_diagnostic_alias_is_rejected_before_parse_error(self):
        with tempfile.TemporaryDirectory(prefix="nano-source-alias-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            text = "fn main( -> int { invalid syntax }"
            source.write_text(text)
            alias = directory / "diagnostics.json"
            os.link(source, alias)
            output = directory / "output.c"
            result = self.invoke([source, "--target", "c", "-o", output,
                                  "--llm-diags-json", alias], directory)
            self.assertEqual(source.read_text(), text)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I will not overwrite", result.stdout + result.stderr)
            self.assertFalse(output.exists())

    def test_uncheckable_destination_identity_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="nano-source-alias-") as tmp:
            directory = Path(tmp)
            source = directory / "input.nano"
            source.write_text(SOURCE)
            alias = directory / "loop"
            alias.symlink_to(alias.name)
            result = self.invoke([source, "--target", "c", "-o", alias], directory)
            self.assertEqual(source.read_text(), SOURCE)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I cannot check", result.stdout + result.stderr)

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
