"""I compare and execute the flat-record emitter subset across VM and AOT."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlatRecordEmitter(unittest.TestCase):
    def run_checked(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_boolean_record_returns_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/flat_record_results.nano"
        with tempfile.TemporaryDirectory(prefix="nano-flat-record-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "make_options", "relay_options", "check_options", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.run_checked(binary)

    def test_void_results_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/void_results.nano"
        with tempfile.TemporaryDirectory(prefix="nano-void-results-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "empty", "show_usage", "explicit_return", "early", "relay", "main")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout,
                                     "usage\ncontinued\n")
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.assertEqual(self.run_checked(binary).stdout, "usage\ncontinued\n")

    def test_scalar_expression_types_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/scalar_expression_types.nano"
        with tempfile.TemporaryDirectory(prefix="nano-scalar-types-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "suffix", "rendered", "check", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.run_checked(binary)

    def test_mixed_scalar_equality_is_refused(self):
        expressions = ['(== true 1)', '(== 1 false)', '(!= true "x")', '(== "x" 1)']
        with tempfile.TemporaryDirectory(prefix="nano-scalar-refusal-") as tmp:
            work = Path(tmp)
            for index, expression in enumerate(expressions):
                with self.subTest(expression=expression):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text('fn main() -> int { assert ' + expression + ' return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_loop_targets_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/loop_control.nano"
        with tempfile.TemporaryDirectory(prefix="nano-loop-targets-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "nested", "exit_outer", "condition_after_continue", "branch_targets", "main")
            self.assertIn("12 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.run_checked(binary)

    def test_loop_control_outside_loop_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-loop-refusal-") as tmp:
            work = Path(tmp)
            for control in ("break", "continue"):
                with self.subTest(control=control):
                    source, output = work / f"{control}.nano", work / f"{control}.nasm"
                    source.write_text('fn before() -> int { while true { break } return 1 }\n'
                                      'fn main() -> int { ' + control + ' return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_wrong_return_result_counts_are_refused(self):
        programs = [
            'fn bad() -> int { return }',
            'fn bad() -> void { return 7 }',
            'fn empty() -> void {} fn bad() -> int { return (empty) }',
            'fn value() -> int { return 7 } fn bad() -> void { return (value) }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-void-refusal-") as tmp:
            work = Path(tmp)
            for index, program in enumerate(programs):
                with self.subTest(program=program):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text(program + '\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_nested_record_results_remain_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-nested-record-") as tmp:
            work = Path(tmp)
            source, output = work / "nested.nano", work / "nested.nasm"
            source.write_text('struct Flags { enabled: bool }\nstruct Nested { child: Flags }\n'
                              'fn make() -> Nested { return Nested { child: Flags { enabled: true } } }\n'
                              'shadow make { let value: Nested = (make) assert value.child.enabled }\n'
                              'fn main() -> int { let value: Nested = (make) assert value.child.enabled return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            self.run_checked(ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", work / "nested.nvm")
            self.run_checked(ROOT / "bin/nano_vm", work / "nested.nvm")
            result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the pinned subset", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
