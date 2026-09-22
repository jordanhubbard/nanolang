"""I compare and execute the flat-record emitter subset across VM and AOT."""
from pathlib import Path
import os
import signal
import shlex
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

    def test_inferred_ordinary_locals_match_and_execute(self):
        fixture = ROOT/'tests/nanoisa/fixtures/inferred_locals.nano'
        with tempfile.TemporaryDirectory(prefix='nano-inferred-locals-') as tmp:
            directory = Path(tmp)
            seed, assembly, emitted = (directory/name for name in ('seed.nvm', 'self.nasm', 'self.nvm'))
            self.run_checked(ROOT/'bin/nano_virt', fixture, '--emit-nvm', '--strip-debug', '-o', seed)
            self.run_checked(ROOT/'bin/nanoisa_emit', fixture, '-o', assembly)
            self.run_checked(ROOT/'tests/nanoisa/test_nanoisa_src_nano', seed, assembly,
                             'once', 'pair', 'fresh', 'values', 'main')
            self.run_checked(ROOT/'bin/nanoisa', 'asm', assembly, '-o', emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT/'bin/nano_vm', '--verify-only', module)
                self.assertEqual(self.run_checked(ROOT/'bin/nano_vm', module).stdout, 'once\n7\n')
                source, native = directory/'native.c', directory/'native'
                self.run_checked(ROOT/'bin/nvm2c', module, '-o', source)
                self.run_checked('cc', '-std=c11', '-Wall', '-Wextra', '-Werror', source, '-lm', '-o', native)
                self.assertEqual(self.run_checked(native).stdout, 'once\n7\n')

    def test_inferred_locals_preserve_ambiguous_and_unsupported_refusals(self):
        cases = {
            'empty array': 'let value = []',
            'empty map': 'let value = (map_new)',
            'unknown initializer': 'let value = missing',
            'explicit mismatch': 'let value: int = 1.5',
        }
        with tempfile.TemporaryDirectory(prefix='nano-inferred-refusals-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'output.nasm'
            for label, statement in cases.items():
                with self.subTest(case=label):
                    source.write_text('fn main() -> int { '+statement+' return 0 }\nshadow main { assert true }\n')
                    output.write_text('retained output')
                    result = subprocess.run([ROOT/'bin/nanoisa_emit', source, '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('I refused', result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'retained output')

    def test_passive_flow_bytecode_and_source_order_metadata_match(self):
        import re
        import struct
        fixture = ROOT / "tests/nanoisa/fixtures/passive_flow.nano"
        with tempfile.TemporaryDirectory(prefix='nano-flow-producers-') as tmp:
            directory = Path(tmp)
            seed, assembly, emitted = (directory/name for name in ('seed.nvm', 'self.nasm', 'self.nvm'))
            self.run_checked(ROOT/'bin/nano_virt', fixture, '--emit-nvm', '--strip-debug', '-o', seed)
            self.run_checked(ROOT/'bin/nanoisa_emit', fixture, '-o', assembly)
            self.run_checked(ROOT/'tests/nanoisa/test_nanoisa_src_nano', seed, assembly,
                         'square', 'diamond', 'scalars', 'multiple', 'shadowed', 'inferred', 'main')
            self.run_checked(ROOT/'bin/nanoisa', 'asm', assembly, '-o', emitted)
            records = []
            for module in (seed, emitted):
                self.run_checked(ROOT/'bin/nano_vm', '--verify-only', module)
                dump = self.run_checked(ROOT/'bin/nanoisa', 'dump', module).stdout
                records.append(bytes.fromhex(''.join(re.findall(r'^\.passive "([0-9a-f]+)"', dump, re.M))))
                self.assertEqual(self.run_checked(ROOT/'bin/nano_vm', module).stdout, '15\n')
                source, native = directory/'native.c', directory/'native'
                self.run_checked(ROOT/'bin/nvm2c', module, '-o', source)
                self.run_checked('cc', '-std=c11', '-Wall', '-Wextra', '-Werror', source, '-lm', '-o', native)
                self.assertEqual(self.run_checked(native).stdout, '15\n')
            self.assertEqual(records[0], records[1])
            words = struct.unpack('<'+'I'*(len(records[0])//4), records[0])
            self.assertEqual(words[:2], (2, 6))
            self.assertEqual(words[2], 2)  # flow record
            self.assertEqual(words[6], 4)  # four source nodes in diamond
            cursor, nodes = 7, []
            for _ in range(4):
                header = words[cursor:cursor+7]
                deps = words[cursor+7:cursor+7+header[3]]
                reads = words[cursor+7+header[3]:cursor+7+header[3]+header[4]]
                nodes.append((header, deps, reads))
                cursor += 7+header[3]+header[4]
            self.assertEqual([node[1] for node in nodes], [(1, 2), (3,), (3,), ()])
            self.assertEqual(nodes[3][2], (0,))
            self.assertEqual(sorted(range(4), key=lambda i: nodes[i][0][0]), [3, 1, 2, 0])

    def test_passive_par_records_and_scalar_execution(self):
        import re
        fixture = ROOT / "tests/nanoisa/fixtures/passive_par.nano"
        with tempfile.TemporaryDirectory(prefix="nano-passive-par-") as tmp:
            directory = Path(tmp)
            seed, assembly, emitted = (directory / name for name in ("seed.nvm", "self.nasm", "self.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly, "independent", "main")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            records = []
            for module in (seed, emitted):
                dumped = self.run_checked(ROOT / "bin/nanoisa", "dump", module).stdout
                record = bytes.fromhex("".join(re.findall(r'^\.passive "([0-9a-f]+)"', dumped, re.MULTILINE)))
                self.assertTrue(record)
                self.assertEqual(int.from_bytes(record[:4], "little"), 2)
                records.append(record)
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "13\n")
                native_c, native = directory / "native.c", directory / "native"
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-lm", "-o", native)
                self.assertEqual(self.run_checked(native).stdout, "13\n")
            self.assertEqual(records[0], records[1])

    def test_passive_original_calculator_scalar_closure(self):
        import re
        from tests.test_passive_par_frontends import calculator_scalar_closure
        with tempfile.TemporaryDirectory(prefix="nano-passive-calculator-") as tmp:
            directory = Path(tmp)
            fixture = directory / "original-closure.nano"
            fixture.write_text(calculator_scalar_closure())
            seed, assembly, emitted = (directory / name for name in ("seed.nvm", "self.nasm", "self.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "int_to_float", "arctan_series", "calculate_pi_machin", "main")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            records = []
            for module in (seed, emitted):
                dumped = self.run_checked(ROOT / "bin/nanoisa", "dump", module).stdout
                records.append(bytes.fromhex("".join(re.findall(r'^\.passive "([0-9a-f]+)"', dumped, re.MULTILINE))))
                self.assertTrue(records[-1])
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "3.14159\n")
                native_c, native = directory / "native.c", directory / "native"
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-lm", "-o", native)
                self.assertEqual(self.run_checked(native).stdout, "3.14159\n")
            self.assertEqual(records[0], records[1])

    def test_executable_closure_preserves_calls_and_refuses_unlowered_roots(self):
        driver_fixture = ROOT / "tests/nanoisa/fixtures/program_closure_driver.nano.txt"
        with tempfile.TemporaryDirectory(prefix="nano-program-closure-") as tmp:
            work = Path(tmp)
            driver, tool = work / "driver.nano", work / "driver"
            driver.write_text(driver_fixture.read_text())
            self.run_checked(ROOT / "bin/nanoc_c", driver, "-o", tool)
            source, assembly, module = work / "program.nano", work / "program.nasm", work / "program.nvm"
            source.write_text(
                'fn dep_a_value(n: int) -> int { if (<= n 0) { return 37 } return (value (- n 1)) }\n'
                'fn dep_a_relay() -> int { return (value 2) }\n'
                'fn dep_a_ping() -> void { (println "A") }\n'
                'fn dep_b_value(n: int) -> int { if (<= n 0) { return 12 } return (value (- n 1)) }\n'
                'fn dep_b_relay() -> int { return (value 2) }\n'
                'fn dep_b_ping() -> void { (println "B") }\n'
                'fn main() -> int { assert (== (values.relay) 37) assert (== (other.relay) 12) '
                'assert (== (alias 2) 37) (values.ping) (other.ping) assert (> (get_argc) 0) return 0 }\n'
                'extern fn unavailable_array_host(path: string) -> array<string>\n'
                'extern fn get_argc() -> int\n'
                'fn unused() -> array<string> { return (unavailable_array_host "unused") }\n')
            output = self.run_checked(tool, source, "bound").stdout
            self.assertEqual(output, self.run_checked(tool, source, "bound").stdout)
            self.assertNotIn("unavailable_array_host", output)
            self.assertNotIn(".function unused", output)
            self.assertIn('.import "" "get_argc" int', output)
            for name in ("dep_a_value", "dep_a_relay", "dep_b_value", "dep_b_relay", "dep_a_ping", "dep_b_ping"):
                self.assertIn(".function " + name, output)
            assembly.write_text(output)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
            self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "A\nB\n")
            native_c, binary = work / "program.c", work / "program"
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
            self.assertEqual(self.run_checked(binary).stdout, "A\nB\n")
            whole = subprocess.run([tool, source, "whole"], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(whole.returncode, 1)
            self.assertEqual(whole.stdout, "")
            # Every initializer is a root, even when main never reads its slot.
            source.write_text('let unused: int = (effect) fn effect() -> int { (println "effect") return 1 } '
                              'let words: array<string> = [] fn main() -> int { assert (== (array_length words) 0) return 0 }\n')
            rooted = self.run_checked(tool, source, "program").stdout
            self.assertIn(".function effect", rooted)
            self.assertIn(".function __init__", rooted)
            self.assertIn("ARR_LITERAL 5 0", rooted)
            assembly.write_text(rooted)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "effect\n")
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
            self.assertEqual(self.run_checked(binary).stdout, "effect\n")
            # Recursive scalar-array globals are roots with their exact child shape.
            source.write_text('let values: array<array<float>> = [[1.5]] '
                              'fn main() -> int { assert (== (at (at values 0) 0) 1.5) return 0 }\n')
            nested = self.run_checked(tool, source, "program").stdout
            self.assertIn("ARR_LITERAL 7 1", nested)
            self.assertIn("STORE_GLOBAL 0", nested)
            assembly.write_text(nested)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", module)
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
            self.run_checked(binary)
            # Same-spelling globals belong to their source modules.
            source.write_text('let count: int = 37\nfn dep_a_relay() -> int { return count }\n\n'
                              'let count: int = 12\nfn dep_b_relay() -> int { return count }\n\n'
                              'fn main() -> int { assert (== (values.relay) 37) '
                              'assert (== (other.relay) 12) return 0 }\n')
            assembly.write_text(self.run_checked(tool, source, "bound").stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", module)
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
            self.run_checked(binary)
            source.write_text('let count: int = 37\nfn dep_a_relay() -> int { return count }\n\n'
                              'let count: int = 12\nfn dep_b_relay() -> int { return count }\n\n'
                              'fn main() -> int { return count }\n')
            wrong_owner = subprocess.run([tool, source, "bound"], cwd=ROOT, capture_output=True,
                                         text=True, timeout=120)
            self.assertEqual(wrong_owner.returncode, 1)
            self.assertEqual(wrong_owner.stdout, "")
            source.write_text('fn target() -> int { return 1 } '
                              'fn consume(f: fn() -> int) -> int { return (f) } '
                              'fn main() -> int { assert (== (consume target) 1) return 0 }\n')
            assembly.write_text(self.run_checked(tool, source, "program").stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
            self.run_checked(ROOT / "bin/nano_vm", module)
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
            self.run_checked(binary)
            refused = [
                'let count: int = 1 fn main() -> int { set count 2 return count }',
                'let count: int = 1 fn __init__() -> void {} fn main() -> int { return count }',
                'extern fn unavailable_array_host(path: string) -> array<string> '
                'fn main() -> array<string> { return (unavailable_array_host "live") }',
                'fn target() -> int { return 1 } let stored: fn() -> int = target '
                'fn main() -> int { return 0 }',
                'fn target() -> int { return 1 } fn main() -> int { let target: int = 0 return (target) }',
            ]
            for program in refused:
                with self.subTest(program=program):
                    source.write_text(program + '\n')
                    result = subprocess.run([tool, source, "program"], cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_globals_initialize_once_and_preserve_mutation(self):
        fixture = ROOT / "tests/nanoisa/fixtures/global_initialization.nano"
        with tempfile.TemporaryDirectory(prefix="nano-globals-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "announce", "change", "local", "main", "__init__")
            self.assertIn("12 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "init\n")
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.assertEqual(self.run_checked(binary).stdout, "init\n")

    def test_scalar_float_values_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/scalar_floats.nano"
        with tempfile.TemporaryDirectory(prefix="nano-scalar-float-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "precise", "negate", "bare", "negative_zero", "difference_zero", "arithmetic",
                             "comparisons", "relay", "change", "render_float", "main", "__init__")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.run_checked(binary)
            binary = work / "reference"
            self.run_checked(ROOT / "bin/nanoc_c", fixture, "-o", binary)
            self.run_checked(binary)

    def test_float_call_return_types_preserve_refusals(self):
        with tempfile.TemporaryDirectory(prefix="nano-float-return-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for declared, actual, value in (("int", "float", "1.0"), ("float", "int", "1")):
                source.write_text(f"fn value() -> {actual} {{ return {value} }} fn main() -> {declared} {{ return (value) }}")
                output.write_text("previous accepted assembly")
                result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_text(), "previous accepted assembly")

    def test_float_format_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/float_format.nano"
        with tempfile.TemporaryDirectory(prefix="nano-scalar-float-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "render_float", "once", "huge", "main", "__init__")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            expected_output = None
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                output = self.run_checked(ROOT / "bin/nano_vm", module).stdout
                if expected_output is None:
                    expected_output = output
                self.assertEqual(output, expected_output)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.assertEqual(self.run_checked(binary).stdout, expected_output)
            binary = work / "reference"
            self.run_checked(ROOT / "bin/nanoc_c", fixture, "-o", binary)
            self.assertEqual(self.run_checked(binary).stdout, expected_output)

    def test_scalar_float_operands_preserve_refusals(self):
        bodies = [
            'return (float_to_string 1.0)',
            'return (+ 1.5 true)', 'return (+ 1.5 "bad")',
            'return (+ 1.5 1)', 'let value: float = 1 return value',
            'let value: int = 1.0 return 1.0',
            'return (% 1.5 1.0)', 'return (and 1.5 1.0)',
            'let text: string = (float_to_string 1) return 1.0',
            'let text: string = (float_to_string 1.0 2.0) return 1.0',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-float-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for body in bodies:
                with self.subTest(body=body):
                    source.write_text('fn main() -> float { ' + body + ' }')
                    output.write_text("previous accepted assembly")
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_text(), "previous accepted assembly")

    def test_range_for_loops_match_scope_and_control_flow(self):
        fixture = ROOT / "tests/nanoisa/fixtures/range_for.nano"
        with tempfile.TemporaryDirectory(prefix="nano-range-for-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "lower", "upper", "bounds", "scoped", "mixed_loops", "first", "scalar_arrays", "main", "__init__")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.run_checked(binary)
            binary = work / "reference"
            self.run_checked(ROOT / "bin/nanoc_c", fixture, "-o", binary)
            self.run_checked(binary)

    def test_range_and_iteration_reject_invalid_operands(self):
        programs = [
            'fn main() -> int { for value in (range) { assert true } return 0 }',
            'fn main() -> int { for value in (range 1 2 3) { assert true } return 0 }',
            'fn main() -> int { for value in (range 0 true) { assert true } return 0 }',
            'fn main() -> int { for value in (range 0 "end") { assert true } return 0 }',
            'fn main() -> int { for value in (range 2) { assert true } return 0 }',
            'fn main() -> int { let values: array<int> = (range 0 2) return 0 }',
            'fn main() -> int { for value in 7 { (println value) } return 0 }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-range-refusal-") as tmp:
            source, output = Path(tmp) / "bad.nano", Path(tmp) / "bad.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program)
                    output.write_text("prior output")
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_text(), "prior output")

    def test_global_filled_arrays_retain_initializer_temporaries(self):
        fixture = ROOT / "tests/nanoisa/fixtures/global_filled_arrays.nano"
        with tempfile.TemporaryDirectory(prefix="nano-global-filled-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.assertIn(".function __init__ 0 12 0 void 0", assembly.read_text())
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "count", "fill", "main", "__init__")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.run_checked(binary)

    def test_aggregate_globals_match_and_execute_in_vm(self):
        fixture = ROOT / "tests/nanoisa/fixtures/global_aggregate_initialization.nano"
        with tempfile.TemporaryDirectory(prefix="nano-aggregate-globals-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "announce", "new_values", "change", "local", "main", "__init__")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "init\n")

    def test_array_access_result_types_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/array_access_types.nano"
        with tempfile.TemporaryDirectory(prefix="nano-array-access-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "words", "joined", "list_label", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_array_access_wrong_container_and_index_are_refused(self):
        programs = [
            'fn main() -> int { return (at 7 0) }',
            'fn main() -> int { let xs: array<int> = [1] return (at xs "zero") }',
            'fn main() -> int { let xs: array<int> = [1] return (at xs) }',
            'fn main() -> int { let xs: List<string> = (list_string_new) return (list_int_get xs 0) }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-array-access-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program + "\n")
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_array_record_fields_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/array_record_fields.nano"
        with tempfile.TemporaryDirectory(prefix="nano-array-record-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "make_result", "forward", "empty", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_wrong_array_record_fields_are_refused(self):
        programs = [
            'struct Box { words: array<string> } fn main() -> Box { return Box { words: [1] } }',
            'struct Box { words: array<string> } fn main() -> Box { let xs: array<int> = [1] return Box { words: xs } }',
            'struct Box { value: int } fn main() -> Box { return Box { value: [1] } }',
            'struct Box { flags: array<array<float>> } fn main() -> Box { return Box { flags: [[1.5]] } }',
            'struct Box { nested: array<array<int>> } fn main() -> Box { return Box { nested: [[1]] } }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-array-record-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program + "\n")
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_computed_array_literal_tags_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/computed_array_literals.nano"
        with tempfile.TemporaryDirectory(prefix="nano-computed-array-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly, "word", "main")
            self.assertIn("6 passed, 0 failed", result.stdout)
            self.assertIn("ARR_LITERAL 5 1", assembly.read_text())
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)
            for expression in ('[(int_to_string 7), 8]', '[8, (int_to_string 7)]', '[1, 1.5]'):
                with self.subTest(expression=expression):
                    invalid = work / "invalid.nano"
                    output = work / "invalid.nasm"
                    invalid.write_text('fn main() -> int { return (array_length ' + expression + ') }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", invalid, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("homogeneous supported array elements", result.stdout)
                    self.assertFalse(output.exists())

    def test_nested_record_lists_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/nested_record_lists.nano"
        with tempfile.TemporaryDirectory(prefix="nano-nested-lists-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "diagnostic", "report", "forward", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_boolean_arrays_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/boolean_arrays.nano"
        with tempfile.TemporaryDirectory(prefix="nano-boolean-arrays-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            comparison = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                          "blank", "values", "filled", "relay", "flags", "main")
            self.assertIn("14 passed, 0 failed", comparison.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.run_checked(binary)

    def test_empty_array_append_infers_nested_types_and_order(self):
        fixture = ROOT / "tests/nanoisa/fixtures/empty_array_append.nano"
        with tempfile.TemporaryDirectory(prefix="nano-empty-append-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "words", "bits", "announce", "main")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.assertEqual(self.run_checked(ROOT / "bin/nano_vm", module).stdout, "first\nsecond\n")
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.assertEqual(self.run_checked(binary).stdout, "first\nsecond\n")

    def test_boolean_arrays_preserve_element_type_refusals(self):
        programs = [
            'fn main() -> int { (array_push (array_push [] \"x\") true) return 0 }',
            'fn main() -> int { let xs: array<int> = [] (array_push xs \"x\") return 0 }',
            'fn main() -> int { let xs: array<bool> = [1] return 0 }',
            'fn main() -> int { let xs: array<bool> = [true, 1] return 0 }',
            'fn main() -> int { let xs: array<bool> = [true] (array_set xs 0 1) return 0 }',
            'fn main() -> int { let xs: array<bool> = [true] (array_push xs 1) return 0 }',
            'fn main() -> int { let xs: array<int> = [1] (array_push xs true) return 0 }',
            'fn main() -> int { let xs: array<string> = ["x"] (array_push xs true) return 0 }',
            'fn main() -> int { let xs: array<bool> = [true] (array_get xs false) return 0 }',
            'fn main() -> int { let xs: List<int> = (list_int_new) (list_string_push xs "x") return 0 }',
            'fn main() -> int { (array_push 1 true) return 0 }',
            'fn main() -> int { let xs: array<bool> = [true] assert (== (at xs 0) 1) return 0 }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-boolean-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program + '\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertFalse(output.exists())

    def test_filled_arrays_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/filled_arrays.nano"
        with tempfile.TemporaryDirectory(prefix="nano-filled-arrays-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            comparison = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                          "count", "fill", "words", "integers", "rows", "modes", "main")
            self.assertIn("16 passed, 0 failed", comparison.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                self.run_checked(binary)

    def test_filled_arrays_refuse_unsupported_operands(self):
        programs = [
            'fn main() -> int { (array_new) return 0 }',
            'fn main() -> int { (array_new 2) return 0 }',
            'fn main() -> int { (array_new 2 1 3) return 0 }',
            'fn main() -> int { (array_new "two" 1) return 0 }',
            'fn main() -> int { (array_new true 1) return 0 }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-filled-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program + '\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertFalse(output.exists())

    def test_filled_arrays_reject_negative_after_both_operands(self):
        with tempfile.TemporaryDirectory(prefix="nano-filled-negative-") as tmp:
            work = Path(tmp)
            source, seed, assembly, emitted = (work / n for n in ("input.nano", "seed.nvm", "emitter.nasm", "emitter.nvm"))
            # The fill assertion checks the ordered effects before the size guard.
            # A file write makes both calls observable even when native abort does not flush stdout.
            marker = work / "fill-ran.txt"
            source.write_text(
                'extern fn file_write(path: string, text: string) -> int\n'
                'fn count(trace: array<int>) -> int { (array_set trace 0 1) return -1 }\n'
                'shadow count { assert true }\n'
                'fn fill(trace: array<int>) -> int { assert (== (at trace 0) 1) '
                'unsafe { assert (== (file_write "' + str(marker) + '" "fill") 0) } return 7 }\n'
                'shadow fill { assert true }\n'
                'fn main() -> int { let trace: array<int> = [0] '
                'let xs: array<int> = (array_new (count trace) (fill trace)) return 0 }\n'
                'shadow main { assert true }\n')
            self.run_checked(ROOT / "bin/nano_virt", source, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", source, "-o", assembly)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary)
                for command in ([ROOT / "bin/nano_vm", module], [binary]):
                    with self.subTest(module=module.name, backend=command[0].name):
                        marker.unlink(missing_ok=True)
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertEqual(result.returncode, 1 if len(command) == 2 else -signal.SIGABRT,
                                         result.stdout + result.stderr)
                        self.assertEqual(marker.read_text(), "fill")

    def test_string_from_char_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_from_char.nano"
        with tempfile.TemporaryDirectory(prefix="nano-from-char-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "character", "joined", "main")
            self.assertIn("8 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_string_from_char_refuses_bad_signatures(self):
        programs = [
            'fn main() -> int { let text: string = (string_from_char) return 0 }',
            'fn main() -> int { let text: string = (string_from_char 65 66) return 0 }',
            'fn main() -> int { let text: string = (string_from_char "65") return 0 }',
            'fn main() -> int { let text: string = (string_from_char true) return 0 }',
            'extern fn vm_string_from_char(code: string) -> string fn main() -> int { return 0 }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-from-char-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program)
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_list_set_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/list_set.nano"
        with tempfile.TemporaryDirectory(prefix="nano-list-set-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "make", "receiver", "index", "replacement", "update", "main")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_list_set_rejects_bad_operands(self):
        operations = [
            '(list_int_set values 0)',
            '(list_int_set values true 7)',
            '(list_int_set values 0 "wrong")',
            '(list_string_set values 0 "wrong")',
            '(array_set 7 0 8)',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-list-set-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for operation in operations:
                with self.subTest(operation=operation):
                    source.write_text('fn main() -> int { let values: array<int> = [1] ' + operation + ' return 0 }')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_list_set_bounds_trap_in_both_backends(self):
        with tempfile.TemporaryDirectory(prefix="nano-list-set-bounds-") as tmp:
            work = Path(tmp)
            source, assembly = work / "bounds.nano", work / "bounds.nasm"
            seed, emitted = work / "seed.nvm", work / "emitted.nvm"
            for index in (-1, 1, 4294967296):
                with self.subTest(index=index):
                    source.write_text('fn main() -> int { let values: List<int> = (list_int_new) '
                                      '(list_int_push values 1) (list_int_set values ' + str(index) + ' 7) return 0 }')
                    self.run_checked(ROOT / "bin/nano_virt", source, "--emit-nvm", "--strip-debug", "-o", seed)
                    self.run_checked(ROOT / "bin/nanoisa_emit", source, "-o", assembly)
                    self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
                    for module in (seed, emitted):
                        native, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                        self.run_checked(ROOT / "bin/nvm2c", module, "-o", native)
                        self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
                        for command in ([ROOT / "bin/nano_vm", module], [binary]):
                            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
                            if command[0] == binary:
                                self.assertEqual(result.returncode, -signal.SIGABRT)
                            else:
                                self.assertEqual(result.returncode, 1)
                                self.assertRegex(result.stdout + result.stderr, r'(?i)(index|bound)')

    def test_enum_array_annotations_reject_other_kinds(self):
        with tempfile.TemporaryDirectory(prefix="nano-enum-array-refusal-") as tmp:
            work = Path(tmp)
            source, output = work / "bad.nano", work / "prior"
            for value in ("true", "Other { value: 1 }"):
                source.write_text('enum Shade { Dark = -2 } struct Other { value: int } '
                                  'fn main() -> int { let items: array<Shade> = [' + value +
                                  '] return 0 } shadow main { assert (== (main) 0) }')
                for compiler in ("nanoc_c", "nano_virt"):
                    with self.subTest(value=value, compiler=compiler):
                        output.write_bytes(b"prior artifact")
                        command = [ROOT / "bin" / compiler, source, "-o", output]
                        if compiler == "nano_virt": command.append("--emit-nvm")
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("E001 TYPE MISMATCH", result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_enum_array_metadata_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/enum_arrays.nano"
        with tempfile.TemporaryDirectory(prefix="nano-enum-array-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            native = work / "seed-native"
            self.run_checked(ROOT / "bin/nanoc_c", fixture, "-o", native)
            self.run_checked(native)
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "values", "read", "main")
            self.assertIn("8 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_enum_values_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/enum_values.nano"
        with tempfile.TemporaryDirectory(prefix="nano-enum-values-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "low", "forward", "state", "values", "entries", "main")
            self.assertIn("14 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

            direct = work / "direct.nano"
            direct.write_text(fixture.read_text().replace('assert (== (array_length current.history) 2)',
                                                         'assert (== (at current.history 1) 17)').replace(
                'assert (== (array_length (values)) 3)', 'assert (== (at (values) 2) 18)'))
            self.run_checked(ROOT / "bin/nano_virt", direct, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", direct, "-o", assembly)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_enum_members_preserve_refusals(self):
        programs = [
            'enum Mode { Low = -3 } fn main() -> int { return Mode.Missing }',
            'enum Mode { Low = -3 } fn main() -> int { let Mode: int = 9 return Mode.Low }',
            'enum Mode { Low = -3 } enum Mode { Low = 4 } fn main() -> Mode { return Mode.Low }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-enum-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program)
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_unsafe_blocks_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/unsafe_blocks.nano"
        with tempfile.TemporaryDirectory(prefix="nano-unsafe-blocks-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "nested", "loops", "host", "scoped", "main")
            self.assertIn("12 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_unsafe_blocks_preserve_refusals(self):
        programs = [
            'fn main() -> int { unsafe { let hidden: int = 7 } return hidden }',
            'fn main() -> int { unsafe { break } return 0 }',
            'extern fn private_host() -> int fn main() -> int { unsafe { return (private_host) } }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-unsafe-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program)
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())

    def test_map_record_fields_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/map_record_fields.nano"
        with tempfile.TemporaryDirectory(prefix="nano-map-fields-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "collect", "forward", "remember", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

            direct = work / "direct.nano"
            direct.write_text(fixture.read_text().replace('(map_put values key value)',
                                                         '(map_put result.visited_set key value)').replace('(map_has values "second")',
                                                         '(map_has original.visited_set "second")').replace(
                '(map_get values "second") 2)\n    assert',
                '(map_get original.visited_set "second") 2)\n    assert'))
            self.run_checked(ROOT / "bin/nanoisa_emit", direct, "-o", assembly)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            self.run_checked(ROOT / "bin/nano_vm", emitted)
            native, binary = work / "direct.c", work / "direct"
            self.run_checked(ROOT / "bin/nvm2c", emitted, "-o", native)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
            self.run_checked(binary)

    def test_record_arrays_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/record_arrays.nano"
        with tempfile.TemporaryDirectory(prefix="nano-record-arrays-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "symbol", "empty", "entries", "table", "main")
            self.assertIn("12 passed, 0 failed", result.stdout)
            self.assertIn("ARR_LITERAL 8 0", assembly.read_text())
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)
            # My emitter retains direct element projection while the C-seed
            # nominal metadata repair remains separately tracked.
            direct = work / "direct.nano"
            direct.write_text(fixture.read_text().replace(
                'let first: Symbol = (at value.symbols 0)\n    assert (== first.name "first")',
                'let first: Symbol = (at value.symbols 0)\n    assert (== (at value.symbols 0).name "first")').replace(
                'let third: Symbol = (at more 2)\n    assert (== third.location.line 3)',
                'assert (== (at more 2).location.line 3)'))
            self.run_checked(ROOT / "bin/nanoisa_emit", direct, "-o", assembly)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            self.run_checked(ROOT / "bin/nano_vm", emitted)
            native, binary = work / "direct.c", work / "direct"
            self.run_checked(ROOT / "bin/nvm2c", emitted, "-o", native)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
            self.run_checked(binary)

    def test_projected_record_array_global_field_executes_in_both_orders(self):
        store = (
            ".function store 1 1 0 void 0\n"
            "LOAD_LOCAL 0\n"
            "AGG_GET 0\n"
            "AGG_GET 0\n"
            "STORE_GLOBAL 0\n"
            "RET\n"
            ".end\n"
        )
        main = (
            ".function main 0 0 0 int 1\n"
            "PUSH_I64 1\n"
            "AGG_PACK 0 0 0 1\n"
            "ARR_LITERAL 8 1\n"
            "AGG_PACK 0 0 0 1\n"
            "AGG_PACK 0 0 0 1\n"
            "CALL store\n"
            "LOAD_GLOBAL 0\n"
            "PUSH_I64 0\n"
            "ARR_GET\n"
            "AGG_GET 0\n"
            "PUSH_I64 1\n"
            "EQ\n"
            "ASSERT\n"
            "PUSH_I64 0\n"
            "RET\n"
            ".end\n"
        )
        with tempfile.TemporaryDirectory(prefix="nano-projected-global-") as tmp:
            work = Path(tmp)
            for label, functions in (("store-first", store + main),
                                     ("main-first", main + store)):
                with self.subTest(order=label):
                    assembly = work / (label + ".nasm")
                    module = work / (label + ".nvm")
                    native_c = work / (label + ".c")
                    binary = work / label
                    assembly.write_text(".entry main\n" + functions)
                    self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", native_c)
                    self.run_checked(
                        *shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc")),
                        "-std=c11", "-O1", "-g", "-fno-omit-frame-pointer",
                        "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                        native_c, "-lm", "-o", binary,
                    )
                    self.run_checked(binary)

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

    def test_is_alnum_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/is_alnum.nano"
        with tempfile.TemporaryDirectory(prefix="nano-is-alnum-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "--imports", "next_code", "classify", "main", "__init__")
            self.assertIn("11 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_string_edges_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_edges.nano"
        with tempfile.TemporaryDirectory(prefix="nano-string-edges-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "text", "prefix", "suffix", "starts", "ends", "main", "__init__")
            self.assertIn("16 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

    def test_string_concat_matches_and_executes(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_concat.nano"
        with tempfile.TemporaryDirectory(prefix="nano-string-int-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "left", "right", "joined", "main", "__init__")
            self.assertIn("12 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

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

    def test_string_concat_keeps_declared_function_and_local_binding(self):
        with tempfile.TemporaryDirectory(prefix="nano-concat-binding-") as tmp:
            work = Path(tmp)
            source, seed, assembly, emitted = (work / n for n in
                ("program.nano", "seed.nvm", "emitter.nasm", "emitter.nvm"))
            source.write_text('fn str_concat(a: string, b: string) -> string { return b } '
                              'shadow str_concat { assert (== (str_concat "left" "right") "right") } '
                              'fn relay() -> string { return (str_concat "left" "right") } '
                              'shadow relay { assert (== (relay) "right") } '
                              'fn main() -> int { assert (== (relay) "right") return 0 } '
                              'shadow main { assert (== (main) 0) }\n')
            refused_seed = subprocess.run([ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", seed],
                                          cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(refused_seed.returncode, 0)
            self.assertIn("Cannot redefine built-in function", refused_seed.stderr)
            # The raw parser API still preserves explicit declarations, rather
            # than silently replacing them with builtin operations.
            self.run_checked(ROOT / "bin/nanoisa_emit", source, "-o", assembly)
            self.assertNotIn("STR_CONCAT", assembly.read_text())
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (emitted,):
                self.run_checked(ROOT / "bin/nano_vm", module)
                native, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
                self.run_checked(binary)
            assembly.unlink()
            source.write_text('fn main() -> string { let str_concat: int = 0 '
                              'return (str_concat "a" "b") }\n')
            refused = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", assembly],
                                     cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse(assembly.exists())

    def test_is_alnum_wrong_arguments_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-is-alnum-refusal-") as tmp:
            work = Path(tmp)
            for index, call in enumerate(['(is_alnum)', '(is_alnum 65 66)', '(is_alnum "A")', '(is_alnum true)']):
                with self.subTest(call=call):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text('fn main() -> bool { return ' + call + ' }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_string_edges_wrong_arguments_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-string-edges-refusal-") as tmp:
            work = Path(tmp)
            for index, call in enumerate([f'({name}{args})' for name in ('str_starts_with', 'str_ends_with')
                                         for args in ('', ' "x"', ' "x" "y" "z"', ' 42 "x"', ' "x" true')]):
                with self.subTest(call=call):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text('fn main() -> bool { return ' + call + ' }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())

    def test_string_concat_wrong_arguments_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-string-int-refusal-") as tmp:
            work = Path(tmp)
            for index, call in enumerate(['(str_concat)', '(str_concat "x")', '(str_concat "x" "y" "z")',
                                          '(str_concat 42 "x")', '(str_concat "x" true)']):
                with self.subTest(call=call):
                    source, output = work / f"bad{index}.nano", work / f"bad{index}.nasm"
                    source.write_text('fn main() -> string { return ' + call + ' }\n')
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

    def test_string_map_global_constructor_executes_in_vm(self):
        # Standalone AOT map globals remain task95796, independently tracked.
        fixture = ROOT / "tests/nanoisa/fixtures/string_map_global.nano"
        with tempfile.TemporaryDirectory(prefix="nano-map-global-") as tmp:
            work = Path(tmp)
            assembly, module = work / "program.nasm", work / "program.nvm"
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.assertIn("HM_NEW 5 5", assembly.read_text())
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
            self.run_checked(ROOT / "bin/nano_vm", module)

    def test_string_map_constructor_contexts_execute_in_selfhost_backends(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_map_contexts.nano"
        with tempfile.TemporaryDirectory(prefix="nano-map-contexts-") as tmp:
            work = Path(tmp)
            assembly, module = work / "program.nasm", work / "program.nvm"
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            self.assertEqual(assembly.read_text().count("HM_NEW 5 5"), 3)
            self.assertNotIn("HM_NEW 5 1", assembly.read_text())
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
            self.run_checked(ROOT / "bin/nano_vm", module)
            native, binary = work / "program.c", work / "program"
            self.run_checked(ROOT / "bin/nvm2c", module, "-o", native)
            self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
            self.run_checked(binary)

    def test_string_string_maps_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/string_string_maps.nano"
        with tempfile.TemporaryDirectory(prefix="nano-map-values-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "fresh", "relay", "put", "has", "read", "wrap", "unwrap", "main")
            self.assertIn("18 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                self.run_checked(ROOT / "bin/nano_vm", module)
                source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                self.run_checked(binary)

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
            'fn bad(m: HashMap<string,string>) -> void { (map_put m "key" 42) }',
            'fn bad(m: HashMap<string,int>) -> HashMap<string,string> { return m }',
            'fn ints() -> HashMap<string,int> { return (map_new) } '
            'fn bad() -> HashMap<string,string> { return (ints) }',
            'fn bad(m: HashMap<string,int>) -> void { let words: HashMap<string,string> = m }',
            'fn consume(m: HashMap<string,string>) -> int { return 0 } '
            'fn bad(m: HashMap<string,int>) -> int { return (consume m) }',
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
            'extern fn vm_is_alnum(code: int) -> int',
            'extern fn vm_is_alnum(code: string) -> bool',
            'extern fn vm_is_alnum() -> bool',
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

    def test_bytecode_emitter_preserves_global_ownership_and_initialization(self):
        with tempfile.TemporaryDirectory(prefix="nano-global-ownership-") as tmp:
            work = Path(tmp)
            compiler, source = work / "emitter.nvm", work / "input.nano"
            native_text, vm_text, module = work / "native.nasm", work / "vm.nasm", work / "output.nvm"
            program = ('let mut trace: int = 0\n'
                       'fn stamp(n: int) -> int { set trace (+ (* trace 10) n) return n }\n'
                       'shadow stamp { assert true }\n'
                       'let first: int = (stamp 1) let second: int = (stamp 2)\n')
            for i in range(40):
                program += (f'fn worker{i}(arg: int) -> int {{ let first: int = arg '
                            'unsafe { let inner: int = first } '
                            'if true { let nested: int = first } return first }\n'
                            f'shadow worker{i} {{ let shadow_local: int = (worker{i} 5) '
                            'assert (== shadow_local 5) }\n')
            program += ('fn main() -> int { assert (== trace 12) assert (== first 1) '
                        'assert (== second 2) assert (== (worker39 5) 5) return 0 }\n'
                        'shadow main { assert (== (main) 0) }\n')
            source.write_text(program)
            self.run_checked(ROOT / "bin/nano_virt", ROOT / "src_nano/nanoisa_emit.nano",
                             "--emit-nvm", "--strip-debug", "-o", compiler)
            self.run_checked(ROOT / "bin/nanoisa_emit", source, "-o", native_text)
            self.run_checked(ROOT / "bin/nano_vm", compiler, "--", source, "-o", vm_text)
            self.assertEqual(vm_text.read_bytes(), native_text.read_bytes())
            self.run_checked(ROOT / "bin/nanoisa", "asm", vm_text, "-o", module)
            self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
            self.run_checked(ROOT / "bin/nano_vm", module)

    def test_nominal_counts_preserve_unused_declarations_and_bounds(self):
        fixture = ROOT / "tests/nanoisa/fixtures/nominal_type_counts.nano"
        with tempfile.TemporaryDirectory(prefix="nano-nominal-counts-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            text = assembly.read_text()
            self.assertTrue(text.startswith(".types 2 2 1\n.entry "), text[:100])
            self.assertIn("AGG_PACK 0 1 0 2", text)
            self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                             "copy_message", "main")
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    dump = self.run_checked(ROOT / "bin/nanoisa", "dump", module).stdout
                    self.assertIn(".types 2 2 1", dump)
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.run_checked(binary)
            assembly.write_text(text.replace(".types 2 2 1", ".types 1 2 1", 1))
            prior = emitted.read_bytes()
            result = subprocess.run([ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("struct_count 1", result.stderr)
            self.assertEqual(emitted.read_bytes(), prior)

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

    def test_nested_record_results_match_and_execute(self):
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
            self.run_checked(ROOT / "bin/nanoisa_emit", source, "-o", output)
            emitted = work / "emitted.nvm"
            self.run_checked(ROOT / "bin/nanoisa", "asm", output, "-o", emitted)
            for module in (work / "nested.nvm", emitted):
                self.run_checked(ROOT / "bin/nano_vm", module)
                native, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                self.run_checked(ROOT / "bin/nvm2c", module, "-o", native)
                self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native, "-o", binary)
                self.run_checked(binary)

    def test_recursive_and_malformed_nested_records_are_refused(self):
        programs = [
            'struct MapField { values: HashMap<int,int> } fn bad(value: MapField) -> MapField { return value }',
            'struct MapField { values: HashMap<string,int> } fn bad() -> MapField { return MapField { values: 7 } }',
            'struct Cycle { next: Cycle } fn identity(value: Cycle) -> Cycle { return value }',
            'struct Cycle { next: List<Cycle> } fn identity(value: Cycle) -> Cycle { return value }',
            'struct Cycle { next: array<Cycle> } fn identity(value: Cycle) -> Cycle { return value }',
            'struct Left { right: Right } struct Right { left: Left } fn identity(value: Left) -> Left { return value }',
            'struct Inner { value: int } struct Outer { inner: Inner } fn bad() -> Outer { return Outer { inner: 7 } }',
            'struct Inner { value: int } struct Other { value: int } struct Outer { inner: Inner } '
            'fn bad() -> Outer { return Outer { inner: Other { value: 7 } } }',
        ]
        with tempfile.TemporaryDirectory(prefix="nano-nested-refusal-") as tmp:
            source, output = Path(tmp) / "input.nano", Path(tmp) / "output.nasm"
            for program in programs:
                with self.subTest(program=program):
                    source.write_text(program + '\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I refused that program:", result.stdout)
                    self.assertFalse(output.exists())



if __name__ == "__main__":
    unittest.main()
