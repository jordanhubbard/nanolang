"""I compare and execute the flat-record emitter subset across VM and AOT."""
from pathlib import Path
import os
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

    def test_escaped_strings_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/escaped_strings.nano"
        with tempfile.TemporaryDirectory(prefix="nano-escaped-strings-") as tmp:
            work = Path(tmp)
            seed, assembly, repeated, emitted = (work / n for n in
                ("seed.nvm", "emitter.nasm", "repeated.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", repeated)
            self.assertEqual(assembly.read_bytes(), repeated.read_bytes())
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "escaped", "unicode_text", "repeated", "unknown_escape", "nul_prefix", "main")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_scalar_array_results_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/scalar_array_results.nano"
        with tempfile.TemporaryDirectory(prefix="nano-array-results-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "empty", "words", "relay", "integers", "fill", "main")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_unsupported_array_results_and_elements_are_refused(self):
        programs = [
            'fn bad() -> array<bool> { return [true] }',
            'fn bad() -> array<array<int>> { return [[1]] }',
            'fn bad() -> array<string> { return [1] }',
            'fn bad() -> array<int> { return ["wrong"] }',
            'fn bad() -> int { let wrong: array<string> = [1] return 0 }',
            'fn bad() -> int { let mut wrong: array<string> = [] set wrong [true] return 0 }',
            'fn take(xs: array<string>) -> int { return 0 } fn bad() -> int { return (take [1]) }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-array-result-refusal-") as tmp:
            work = Path(tmp)
            for index, program in enumerate(programs):
                with self.subTest(program=program):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text(program + '\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_string_to_int_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_to_int.nano"
        with tempfile.TemporaryDirectory(prefix="nano-string-int-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "convert", "sum", "main")
            self.assertIn("8 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_string_to_int_wrong_arguments_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-string-int-refusal-") as tmp:
            work = Path(tmp)
            for index, call in enumerate(['(string_to_int)', '(string_to_int "1" "2")',
                                          '(string_to_int 42)', '(string_to_int true)']):
                with self.subTest(call=call):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text('fn main() -> int { return ' + call + ' }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_bound_parser_calls_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/bound_calls_reference.nano"
        driver_fixture = ROOT / "tests/nanoisa/fixtures/bound_calls_driver.nano.txt"
        with tempfile.TemporaryDirectory(prefix="nano-bound-calls-") as tmp:
            work = Path(tmp)
            driver, tool = work / "driver.nano", work / "driver"
            driver.write_text(driver_fixture.read_text())
            self.run_checked(ROOT / "bin/nanoc_c", driver, "-o", tool)
            assembly, seed, emitted = work / "bound.nasm", work / "seed.nvm", work / "bound.nvm"
            assembly.write_text(self.run_checked(tool).stdout)
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "dep_a_value", "dep_a_relay", "dep_a_ping", "dep_b_value", "dep_b_relay", "dep_b_ping", "main")
            self.assertIn("16 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "A\nB\n")
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.assertEqual(self.run_checked(binary).stdout, "A\nB\n")

    def test_string_int_maps_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_int_maps.nano"
        with tempfile.TemporaryDirectory(prefix="nano-map-values-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "fresh", "relay", "put", "add", "has", "read", "main")
            self.assertIn("16 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_unsupported_map_shapes_and_arguments_are_refused(self):
        programs = [
            'fn bad() -> HashMap<int,int> { return (map_new) }',
            'fn bad() -> HashMap<string,string> { return (map_new) }',
            'fn bad() -> HashMap<string,bool> { return (map_new) }',
            'fn bad(m: HashMap<string,int>) -> void { (map_put m "key" true) }',
            'fn bad(m: HashMap<string,int>) -> bool { return (map_has m 1) }',
            'fn bad() -> int { return (map_get 0 "key") }',
            'fn bad() -> HashMap<string,int> { return (map_new 1) }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-map-refusal-") as tmp:
            work = Path(tmp)
            for index, program in enumerate(programs):
                with self.subTest(program=program):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text(program + '\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_host_imports_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/host_imports.nano"
        with tempfile.TemporaryDirectory(prefix="nano-host-imports-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "--imports", "env_text", "declared_text", "argument", "argc_value", "present", "main")
            self.assertIn("15 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                for command in ([ROOT / "bin/nano_vm", module, "--", "sentinel"], [binary, "sentinel"]):
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
                                            env={**os.environ, "NANO_EMITTER_HOST_TEST": "host-value"})
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, "host-value\n2\nsentinel\n")

    def test_host_signature_mismatches_are_refused(self):
        programs = [
            'extern fn get_argv(index: string) -> string',
            'extern fn get_argc(unexpected: int) -> int',
            'extern fn get_argc() -> string',
            'extern fn unknown_host() -> int',
            'extern fn get_argv(index: array<int>) -> string',
            'fn bad() -> string { return (getenv 42) }',
            'extern fn get_argv(index: int) -> string fn bad() -> string { return (get_argv "x") }',
            'fn bad() -> string { return (getenv) }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-host-refusal-") as tmp:
            work = Path(tmp)
            for index, program in enumerate(programs):
                with self.subTest(program=program):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text(program + '\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

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
