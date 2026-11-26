"""I execute actual emitted scalar expressions against independent and VM results."""
from pathlib import Path
import itertools
import os
import random
import shlex
import unittest
import tempfile

import yaml

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_module_verifier import assembled
from tests.test_nanolang_translator_stack_heights import ins

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = yaml.safe_load((ROOT / 'spec/nanoisa.yaml').read_text())
OPS = {entry['name']: entry for entry in SCHEMA['legacy_opcodes']}
NAMES = ('I64_ADD', 'I64_SUB', 'I64_MUL', 'I64_DIV_S', 'I64_REM_S', 'I64_NEG',
         'I64_EQ', 'I64_NE', 'I64_LT_S', 'I64_LE_S', 'I64_GT_S', 'I64_GE_S',
         'BOOL_AND', 'BOOL_OR', 'BOOL_NOT')
ACCEPTED = {OPS[name]['code']: name for name in NAMES}
BOOLEAN_RESULTS = frozenset(('I64_EQ', 'I64_NE', 'I64_LT_S', 'I64_LE_S', 'I64_GT_S', 'I64_GE_S', 'BOOL_AND', 'BOOL_OR', 'BOOL_NOT'))
ARITY = {OPS[name]['code']: OPS[name]['pops'] for name in NAMES}
MASK = (1 << 64) - 1
MINIMUM = -(1 << 63)
SIGNED = (MINIMUM, MINIMUM + 1, -(1 << 31), -7, -1, 0, 1, 2, 7,
          (1 << 31) - 1, (1 << 63) - 1)


def signed(bits):
    bits &= MASK
    return bits if bits < 1 << 63 else bits - (1 << 64)


def examples(name):
    if name == 'BOOL_NOT':
        return [(0, 0), (1, 0)]
    if name.startswith('BOOL_'):
        return list(itertools.product((0, 1), repeat=2))
    if name == 'I64_NEG':
        return [(value, 0) for value in SIGNED]
    rng = random.Random(1142)
    return list(itertools.product(SIGNED, repeat=2)) + [
        (signed(rng.getrandbits(64)), signed(rng.getrandbits(64))) for _ in range(8)]


def expected(name, left, right):
    if name == 'I64_ADD': return signed(left + right)
    if name == 'I64_SUB': return signed(left - right)
    if name == 'I64_MUL': return signed(left * right)
    if name == 'I64_NEG': return signed(-left)
    if name in ('I64_DIV_S', 'I64_REM_S'):
        if right == 0: return 0
        if left == MINIMUM and right == -1:
            return MINIMUM if name == 'I64_DIV_S' else 0
        quotient = abs(left) // abs(right)
        if (left < 0) != (right < 0): quotient = -quotient
        return quotient if name == 'I64_DIV_S' else left - quotient * right
    if name == 'I64_EQ': return int(left == right)
    if name == 'I64_NE': return int(left != right)
    if name == 'I64_LT_S': return int(left < right)
    if name == 'I64_LE_S': return int(left <= right)
    if name == 'I64_GT_S': return int(left > right)
    if name == 'I64_GE_S': return int(left >= right)
    if name == 'BOOL_AND': return int(bool(left) and bool(right))
    if name == 'BOOL_OR': return int(bool(left) or bool(right))
    if name == 'BOOL_NOT': return int(not left)
    raise AssertionError(name)


def controls():
    rows = []
    for opcode in range(-1, 257):
        rows.append(('base', opcode, [0] if opcode in (197, 226) else [0, 1]))
    for name in NAMES:
        opcode, arity = OPS[name]['code'], OPS[name]['pops']
        rows.extend((f'arity-{count}', opcode, list(range(count))) for count in range(4))
        if arity == 1:
            rows.extend((label, opcode, [value]) for label, value in (
                ('left-low', -1), ('left-high', 1 << 32), ('left-edge', MASK >> 32)))
        else:
            rows.extend((label, opcode, slots) for label, slots in (
                ('left-low', [-1, 1]), ('left-high', [1 << 32, 1]),
                ('left-edge', [MASK >> 32, 1]), ('right-low', [0, -1]),
                ('right-high', [0, 1 << 32]), ('right-edge', [0, MASK >> 32]),
                ('alias', [5, 5])))
    return rows


def execution_source(prelude, valid):
    lines = [prelude, '#include <stdio.h>', '#include <inttypes.h>',
             '#include <string.h>', '#include <limits.h>',
             '_Static_assert(CHAR_BIT == 8, "I require eight-bit bytes");',
             'static int64_t input_bits(uint64_t bits) {',
             ' int64_t value; memcpy(&value, &bits, sizeof value); return value;', '}']
    inputs, outputs = [], []
    for index, opcode, slots, expression in valid:
        lines.extend((f'static int64_t scalar_case_{index}(int64_t left, int64_t right) {{',
                      ' (void)left; (void)right;',
                      f' int64_t nano_aot_local_{slots[0]} = left;'))
        if len(slots) == 2 and slots[0] != slots[1]:
            lines.append(f' int64_t nano_aot_local_{slots[1]} = right;')
        lines.extend((f' return {expression};', '}'))
        name = ACCEPTED[opcode]
        for left, right in examples(name):
            inputs.append(f'{index} {left & MASK:016x} {right & MASK:016x}\n')
            effective_right = left if len(slots) == 2 and slots[0] == slots[1] else right
            outputs.append(f'{expected(name, left, effective_right) & MASK:016x}\n')
    lines.extend(('int main(int argc, char **argv) {',
                  ' if (argc != 2) return 2;',
                  ' FILE *file = fopen(argv[1], "r"); if (!file) return 2;',
                  ' unsigned index; uint64_t left, right; int scanned;',
                  ' while ((scanned = fscanf(file, "%u %" SCNx64 " %" SCNx64, &index, &left, &right)) == 3) {',
                  '  int64_t result;', '  switch (index) {'))
    lines.extend(f'   case {index}: result = scalar_case_{index}(input_bits(left), input_bits(right)); break;'
                 for index, _, _, _ in valid)
    lines.extend(('   default: fclose(file); return 3;', '  }',
                  '  if (printf("%016" PRIx64 "\\n", (uint64_t)result) < 0) { fclose(file); return 4; }',
                  ' }', ' int failed = scanned != EOF || ferror(file);',
                  ' if (fclose(file)) failed = 1;', ' return failed ? 5 : 0;', '}'))
    return '\n'.join(lines) + '\n', ''.join(inputs), ''.join(outputs)


class ScalarC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-scalar-c-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        cls.fixture = ROOT / 'tests/nvm2c_scalar_c_driver.nano'
        cls.routes = build_routes(cls.fixture, cls.directory)

    def test_reports_and_actual_emitted_c(self):
        original_report = None
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        cases = controls()
        self.assertEqual(len(ACCEPTED), 15)
        self.assertEqual(len(cases), 415)
        for route_name, route in self.routes:
            with self.subTest(route=route_name):
                report = checked(route)
                if original_report is None: original_report = report
                self.assertEqual(report, original_report)
                begin, prelude, remaining = report.partition('PRELUDE-BEGIN\n')[0], '', ''
                self.assertEqual(begin, '')
                prelude, marker, remaining = report[len('PRELUDE-BEGIN\n'):].partition('PRELUDE-END\n')
                self.assertEqual(marker, 'PRELUDE-END\n')
                self.assertTrue(prelude.startswith('#include <stdint.h>\n'))
                rows = [line.split('|', 7) for line in remaining.splitlines()]
                self.assertEqual(len(rows), len(cases))
                valid = []
                for index, (row, (label, opcode, slots)) in enumerate(zip(rows, cases)):
                    self.assertEqual(row[:4], ['E', label, str(opcode), ','.join(map(str, slots))])
                    accepted = opcode in ACCEPTED and len(slots) == ARITY[opcode] and all(0 <= n < 1 << 32 for n in slots)
                    self.assertEqual(row[4], str(int(accepted)))
                    if not accepted:
                        self.assertEqual(row[5:], ['0', '0', ''])
                        continue
                    name = ACCEPTED[opcode]
                    self.assertEqual(row[5:7], [str(ARITY[opcode]), str(int(name in BOOLEAN_RESULTS))])
                    self.assertTrue(row[7])
                    valid.append((index, opcode, slots, row[7]))
                self.assertEqual(len(valid), 71)
                source, inputs, wanted = execution_source(prelude, valid)
                c_path = self.directory / f'{route_name}-emitted.c'
                data_path = self.directory / f'{route_name}-inputs.txt'
                c_path.write_text(source)
                data_path.write_text(inputs)
                for mode, flags in [('sanitized', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']),
                                    ('optimized', ['-O3'])]:
                    binary = self.directory / f'{route_name}-{mode}'
                    checked([*compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', *flags,
                             c_path, *native_link_flags(), '-o', binary], build=True)
                    self.assertEqual(checked([binary, data_path]), wanted)

    def test_actual_vm_and_reference_c(self):
        code = bytearray()
        wanted = []
        for name in NAMES:
            for left, right in examples(name):
                push = 'PUSH_BOOL' if name.startswith('BOOL_') else 'PUSH_I64'
                code.extend(ins(push, left))
                if OPS[name]['pops'] == 2: code.extend(ins(push, right))
                code.extend(ins(name) + ins('CAST_INT') + ins('PRINTLN'))
                wanted.append(str(expected(name, left, right)) + '\n')
        code.extend(ins('PUSH_I64', 0) + ins('RET'))
        module = self.directory / 'scalar-reference.nvm'
        module.write_bytes(assembled([bytes(code)], returns=[[1]], declared=[2], strings=(b'scalar_reference',)))
        checked([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.assertEqual(checked([ROOT / 'bin/nano_vm', module, '--']), ''.join(wanted))
        c_path = self.directory / 'scalar-reference.c'
        binary = self.directory / 'scalar-reference-native'
        checked([ROOT / 'bin/nvm2c', module, '-o', c_path], build=True)
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        platform = ['-ldl', '-Wl,--export-dynamic'] if os.uname().sysname == 'Linux' else []
        checked([*compiler, '-std=c11', '-O1', '-g', '-fsanitize=address,undefined',
                 '-fno-sanitize-recover=all', c_path, ROOT / 'bin/nano_aot_runtime.o',
                 '-lm', *platform, *native_link_flags(), '-o', binary], build=True)
        self.assertEqual(checked([binary]), ''.join(wanted))

    def test_imported_shadows(self):
        native = self.directory / 'imported-shadows-native'
        checked([ROOT / 'bin/nanoc_c', self.fixture, '--test-imports', '-o', native], build=True)
        self.assertTrue(checked([native]).startswith('PRELUDE-BEGIN\n'))
        producers = [('c', [ROOT / 'bin/nano_virt'])]
        if source_compiler_command(): producers.append(('nanolang', source_compiler_command()))
        for name, producer in producers:
            module = self.directory / f'imported-shadows-{name}.nvm'
            checked([*producer, self.fixture, '--test-imports', '--emit-nvm', '-o', module], build=True)
            checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.assertTrue(checked([ROOT / 'bin/nano_vm', module, '--']).startswith('PRELUDE-BEGIN\n'))
