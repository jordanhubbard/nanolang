"""I execute my actual binary64 C text against exact bit/rational expectations."""
from pathlib import Path
from fractions import Fraction
import itertools
import os
import random
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_module_verifier import assembled
from tests.test_nanolang_translator_stack_heights import ins
from tests.test_nanolang_translator_scalar_c import OPS, MASK, signed

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('F64_FROM_BITS', 'F64_TO_BITS', 'F64_ADD', 'F64_SUB', 'F64_MUL', 'F64_DIV',
         'F64_NEG', 'F64_EQ', 'F64_NE', 'F64_LT', 'F64_LE', 'F64_GT', 'F64_GE')
ACCEPTED = {OPS[n]['code']: n for n in NAMES}
ARITY = {OPS[n]['code']: OPS[n]['pops'] for n in NAMES}
BOOLEAN = frozenset(('F64_EQ', 'F64_NE', 'F64_LT', 'F64_LE', 'F64_GT', 'F64_GE'))
SIGN, INF, NAN, ONE = 1 << 63, 0x7ff0000000000000, 0x7ff8000000000000, 0x3ff0000000000000
PATTERNS = (0, SIGN, 1, SIGN | 1, 3, 0x000fffffffffffff, 0x0010000000000000,
            ONE, ONE + 1, ONE | SIGN, 0x3fe0000000000000, 0x3feffffffffffffe,
            0x3ca0000000000000, 0x4008000000000000, 0x7fefffffffffffff,
            0xffefffffffffffff, INF, INF | SIGN, 0x7ff0000000000001,
            0xfff0000000000042, 0x7ff8123456789abc, 0xfff8000000001234)


def fraction(bits):
    exponent, mantissa = (bits >> 52) & 2047, bits & ((1 << 52) - 1)
    assert exponent != 2047
    if exponent: mantissa |= 1 << 52
    power = exponent - 1075 if exponent else -1074
    value = Fraction(mantissa << max(power, 0), 1 << max(-power, 0))
    return -value if bits & SIGN else value


def rounded(value, zero_sign=0):
    """I round a rational to binary64 using integer ties-to-even arithmetic."""
    if not value: return zero_sign
    sign = SIGN if value < 0 else 0
    value = abs(value)
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d): exponent -= 1
    quantum = max(exponent - 52, -1074)
    numerator, denominator = n << max(-quantum, 0), d << max(quantum, 0)
    significand, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and significand & 1): significand += 1
    if exponent < -1022: return sign | significand
    if significand == 1 << 53: significand >>= 1; exponent += 1
    if exponent > 1023: return sign | INF
    return sign | ((exponent + 1023) << 52) | (significand - (1 << 52))


def expected(name, left, right):
    if name in ('F64_FROM_BITS', 'F64_TO_BITS'): return left
    if name == 'F64_NEG': return left ^ SIGN
    nan_left, nan_right = (left & ~SIGN) > INF, (right & ~SIGN) > INF
    if name in BOOLEAN:
        if nan_left or nan_right: return int(name == 'F64_NE')
        equal = left == right or ((left & ~SIGN) == 0 and (right & ~SIGN) == 0)
        key = lambda bits: (MASK ^ bits) if bits & SIGN else (bits | SIGN)
        less = not equal and key(left) < key(right)
        return {'F64_EQ': int(equal), 'F64_NE': int(not equal), 'F64_LT': int(less),
                'F64_LE': int(less or equal), 'F64_GT': int(not less and not equal),
                'F64_GE': int(not less)}[name]
    if name == 'F64_DIV' and (right & ~SIGN) == 0: return 0
    if nan_left or nan_right: return NAN
    if name == 'F64_SUB': right ^= SIGN; name = 'F64_ADD'
    inf_left, inf_right = (left & ~SIGN) == INF, (right & ~SIGN) == INF
    zero_left, zero_right = (left & ~SIGN) == 0, (right & ~SIGN) == 0
    sign = (left ^ right) & SIGN
    if name == 'F64_ADD':
        if inf_left and inf_right: return NAN if sign else left
        if inf_left or inf_right: return left if inf_left else right
        return rounded(fraction(left) + fraction(right), left & right & SIGN)
    if name == 'F64_MUL':
        if (inf_left and zero_right) or (inf_right and zero_left): return NAN
        if inf_left or inf_right: return sign | INF
        return rounded(fraction(left) * fraction(right), sign)
    assert name == 'F64_DIV'
    if inf_left and inf_right: return NAN
    if inf_left: return sign | INF
    if inf_right: return sign
    return rounded(fraction(left) / fraction(right), sign)


def examples(name):
    rng = random.Random(0xf641142)
    if OPS[name]['pops'] == 1:
        return [(x, 0) for x in dict.fromkeys((*PATTERNS, *(1 << n for n in range(64)),
                                             *(rng.getrandbits(64) for _ in range(32))))]
    return [*itertools.product(PATTERNS, repeat=2),
            *((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(32))]


def controls():
    rows = [('base', op, [0] if op in (141, 142, 204) else [0, 1]) for op in range(-1, 257)]
    for name in NAMES:
        op, arity = OPS[name]['code'], OPS[name]['pops']
        rows.extend((f'arity-{n}', op, list(range(n))) for n in range(4))
        if arity == 1:
            rows.extend((label, op, [value]) for label, value in (
                ('left-low', -1), ('left-high', 1 << 32), ('left-edge', (1 << 32) - 1)))
        else:
            rows.extend((label, op, slots) for label, slots in (
                ('left-low', [-1, 1]), ('left-high', [1 << 32, 1]), ('left-edge', [(1 << 32) - 1, 1]),
                ('right-low', [0, -1]), ('right-high', [0, 1 << 32]), ('right-edge', [0, (1 << 32) - 1]),
                ('alias', [5, 5])))
    return rows


def execution_source(prelude, valid):
    lines = [prelude, '#include <stdio.h>', '#include <inttypes.h>',
             'static int64_t input_i64(uint64_t bits) { int64_t value; memcpy(&value, &bits, 8); return value; }',
             'static double input_f64(uint64_t bits) { double value; memcpy(&value, &bits, 8); return value; }']
    inputs, wanted = [], []
    for index, op, slots, text in valid:
        name = ACCEPTED[op]
        kind = 'int64_t' if name == 'F64_FROM_BITS' else 'double'
        value = 'input_i64' if kind == 'int64_t' else 'input_f64'
        lines.extend((f'static uint64_t float_case_{index}(uint64_t left, uint64_t right) {{',
                      ' (void)right;', f' {kind} nano_aot_local_{slots[0]} = {value}(left);'))
        if len(slots) == 2 and slots[0] != slots[1]:
            lines.append(f' double nano_aot_local_{slots[1]} = input_f64(right);')
        if name in BOOLEAN or name == 'F64_TO_BITS': lines.append(f' return (uint64_t){text};')
        else: lines.extend((f' double result = {text}; uint64_t bits;', ' memcpy(&bits, &result, 8); return bits;'))
        lines.append('}')
        for left, right in examples(name):
            inputs.append(f'{index} {left:016x} {right:016x}\n')
            actual_right = left if len(slots) == 2 and slots[0] == slots[1] else right
            wanted.append(f'{expected(name, left, actual_right):016x}\n')
    lines.extend(('int main(int argc, char **argv) {',
                  ' if (argc != 2) return 2;', ' FILE *file = fopen(argv[1], "r"); if (!file) return 2;',
                  ' unsigned index; uint64_t left, right, result; int scanned;',
                  ' while ((scanned = fscanf(file, "%u %" SCNx64 " %" SCNx64, &index, &left, &right)) == 3) {',
                  '  switch (index) {'))
    lines.extend(f'   case {index}: result = float_case_{index}(left, right); break;' for index, _, _, _ in valid)
    lines.extend(('   default: fclose(file); return 3;', '  }',
                  '  if (printf("%016" PRIx64 "\\n", result) < 0) { fclose(file); return 4; }',
                  ' }', ' int failed = scanned != EOF || ferror(file);',
                  ' if (fclose(file)) failed = 1;', ' return failed ? 5 : 0;', '}'))
    return '\n'.join(lines) + '\n', ''.join(inputs), ''.join(wanted)


class Binary64C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-binary64-c-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        cls.fixture = ROOT / 'tests/nvm2c_binary64_c_driver.nano'
        cls.routes = build_routes(cls.fixture, cls.directory)

    def test_exact_oracle_named_controls(self):
        cases = [('F64_ADD', INF, INF | SIGN, NAN), ('F64_SUB', INF, INF, NAN),
                 ('F64_MUL', 0, INF, NAN), ('F64_DIV', INF, INF, NAN),
                 ('F64_ADD', SIGN, SIGN, SIGN), ('F64_ADD', SIGN, 0, 0),
                 ('F64_MUL', SIGN, ONE, SIGN), ('F64_ADD', ONE, 0x3ca0000000000000, ONE),
                 ('F64_ADD', ONE + 1, 0x3ca0000000000000, ONE + 2),
                 ('F64_MUL', 1, 0x3fe0000000000000, 0), ('F64_MUL', 3, 0x3fe0000000000000, 2),
                 ('F64_MUL', SIGN | 1, 0x3fe0000000000000, SIGN), ('F64_ADD', 1, 1, 2),
                 ('F64_ADD', 0x7fefffffffffffff, 0x7fefffffffffffff, INF),
                 ('F64_DIV', ONE, 0x4008000000000000, 0x3fd5555555555555)]
        cases += [('F64_DIV', left, right, 0) for left in PATTERNS for right in (0, SIGN)]
        cases += [('F64_NEG', left, 0, left ^ SIGN) for left in PATTERNS]
        cases += [(name, left, left, int(name == 'F64_NE')) for left in PATTERNS if (left & ~SIGN) > INF for name in BOOLEAN]
        for name, left, right, result in cases:
            self.assertEqual(expected(name, left, right), result, (name, hex(left), hex(right)))

    def test_actual_vm_and_reference_c(self):
        code, wanted = bytearray(), []
        for name in NAMES:
            for left, right in examples(name):
                code.extend(ins('PUSH_I64', signed(left)) if name == 'F64_FROM_BITS' else bytes([OPS['PUSH_F64']['code']]) + struct.pack('<Q', left))
                if OPS[name]['pops'] == 2: code.extend(bytes([OPS['PUSH_F64']['code']]) + struct.pack('<Q', right))
                code.extend(ins(name))
                if name not in BOOLEAN and name != 'F64_TO_BITS': code.extend(ins('F64_TO_BITS'))
                code.extend(ins('CAST_INT') + ins('PRINTLN'))
                wanted.append(str(signed(expected(name, left, right))) + '\n')
        code.extend(ins('PUSH_I64', 0) + ins('RET'))
        module, c_path, binary = (self.directory / n for n in ('float-reference.nvm', 'float-reference.c', 'float-reference'))
        module.write_bytes(assembled([bytes(code)], returns=[[1]], declared=[2], strings=(b'float_reference',)))
        checked([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.assertEqual(checked([ROOT / 'bin/nano_vm', module, '--']), ''.join(wanted))
        checked([ROOT / 'bin/nvm2c', module, '-o', c_path], build=True)
        platform = ['-ldl', '-Wl,--export-dynamic'] if os.uname().sysname == 'Linux' else []
        checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all', c_path,
                 ROOT / 'bin/nano_aot_runtime.o', '-lm', *platform, *native_link_flags(), '-o', binary], build=True)
        self.assertEqual(checked([binary]), ''.join(wanted))

    def test_reports_and_actual_emitted_c(self):
        original = None
        cases = controls()
        self.assertEqual(len(cases), 389)
        for route_name, route in self.routes:
            with self.subTest(route=route_name):
                report = checked(route)
                if original is None: original = report
                self.assertEqual(report, original)
                self.assertTrue(report.startswith('PRELUDE-BEGIN\n'))
                prelude, marker, rest = report[len('PRELUDE-BEGIN\n'):].partition('PRELUDE-END\n')
                self.assertEqual(marker, 'PRELUDE-END\n')
                rows = [line.split('|', 7) for line in rest.splitlines()]
                self.assertEqual(len(rows), len(cases))
                valid = []
                for index, (row, (label, op, slots)) in enumerate(zip(rows, cases)):
                    self.assertEqual(row[:4], ['E', label, str(op), ','.join(map(str, slots))])
                    accepted = op in ACCEPTED and len(slots) == ARITY[op] and all(0 <= n < 1 << 32 for n in slots)
                    self.assertEqual(row[4], str(int(accepted)))
                    if not accepted: self.assertEqual(row[5:], ['0', '0', '']); continue
                    self.assertEqual(row[5:7], [str(ARITY[op]), str(int(ACCEPTED[op] in BOOLEAN))])
                    self.assertTrue(row[7]); valid.append((index, op, slots, row[7]))
                self.assertEqual(len(valid), 59)
                source, inputs, wanted = execution_source(prelude, valid)
                c_path, data_path = self.directory / (route_name + '.c'), self.directory / (route_name + '.txt')
                c_path.write_text(source); data_path.write_text(inputs)
                modes = [('sanitized', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']),
                         ('optimized', ['-O3'])]
                for mode, flags in modes:
                    binary = self.directory / (route_name + '-' + mode)
                    checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra', '-Werror',
                             *flags, '-I', ROOT / 'src', c_path, *native_link_flags(), '-o', binary], build=True)
                    self.assertEqual(checked([binary, data_path]), wanted)

    def test_imported_shadows(self):
        binary = self.directory / 'imported-native'
        checked([ROOT / 'bin/nanoc_c', self.fixture, '--test-imports', '-o', binary], build=True)
        self.assertTrue(checked([binary]).startswith('PRELUDE-BEGIN\n'))
        producers = [('c', [ROOT / 'bin/nano_virt'])]
        if source_compiler_command(): producers.append(('nanolang', source_compiler_command()))
        for name, command in producers:
            module = self.directory / ('imported-' + name + '.nvm')
            checked([*command, self.fixture, '--test-imports', '--emit-nvm', '-o', module], build=True)
            checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.assertTrue(checked([ROOT / 'bin/nano_vm', module, '--']).startswith('PRELUDE-BEGIN\n'))
