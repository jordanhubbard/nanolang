"""I reconstruct typed arithmetic with exact bits and explicitly selected fresh tools."""
import os
from pathlib import Path
import unittest
from tests import test_reconstruction_f64_transport as transport
from tests import test_reconstruction_f64_comparisons as comparisons
from scripts.nanoisa_reconstruction import Analyze, Expr, INT, FLOAT, BOOL, Refusal, FLOAT_ARITHMETIC

ROOT = Path(__file__).resolve().parents[1]


def signed(bits):
    return bits if bits < 1 << 63 else bits - (1 << 64)


class FloatArithmetic(unittest.TestCase):
    command = transport.FloatReconstruction.command
    module = transport.FloatReconstruction.module
    native = transport.FloatReconstruction.native
    paired = transport.FloatReconstruction.paired
    inspect = comparisons.FloatComparisons.inspect
    observe_call_count = transport.FloatReconstruction.observe_call_count

    def setUp(self):
        transport.FloatReconstruction.setUp(self)
        # I select this checkout's fresh bootstrap explicitly, not a historical
        # inherited NANO_HL_COMPILER_DIR. Qualification records these tool hashes.
        self.assertEqual(transport.TOOLS.resolve(), (ROOT / 'bin').resolve(),
                         'I require this checkout\'s fresh post-policy producers')

    def test_exact_results_and_preserved_inputs(self):
        one = 0x3ff0000000000000
        two = 0x4000000000000000
        infinity = 0x7ff0000000000000
        nan = 0x7ff8000000000000
        rows = {
            'F64_ADD': [(one, one, two), (one, 0x3ca0000000000000, one),
                        (one + 1, 0x3ca0000000000000, one + 2),
                        (0x7fefffffffffffff, 0x7fefffffffffffff, infinity),
                        (infinity, 0xfff0000000000000, nan),
                        (0x8000000000000000, 0x8000000000000000, 0x8000000000000000),
                        (0x000fffffffffffff, 1, 0x0010000000000000)],
            'F64_SUB': [(two, one, one), (one, two, 0xbff0000000000000),
                        (one, one, 0), (infinity, infinity, nan),
                        (0x0010000000000000, 0x000fffffffffffff, 1)],
            'F64_MUL': [(one, two, two), (0, infinity, nan),
                        (1, 0x3fe0000000000000, 0),
                        (3, 0x3fe0000000000000, 2),
                        (0x0010000000000000, 0x3fe0000000000000, 0x0008000000000000),
                        (0x8000000000000000, two, 0x8000000000000000)],
            'F64_DIV': [(two, two, one), (one, 0x4008000000000000, 0x3fd5555555555555),
                        (infinity, infinity, nan), (1, two, 0), (3, two, 2)],
        }
        for op in FLOAT_ARITHMETIC:
            for payload in (0x7ff8000000000042, 0xfff0000000000043):
                rows[op].extend(((payload, one, nan), (one, payload, nan)))
        for divisor in (0, 0x8000000000000000):
            for numerator in (one, 0xbff0000000000000, infinity,
                              0xfff0000000000000, 0x7ff8000000000042,
                              0xfff0000000000043):
                rows['F64_DIV'].append((numerator, divisor, 0))
        for op, cases in rows.items():
            body = ''
            ordinal = 0
            for left, right, result in cases:
                body += f'PUSH_F64 bits:{left:016x}\nSTORE_LOCAL 0\nPUSH_F64 bits:{right:016x}\nSTORE_LOCAL 1\n'
                for expression, expected in ((f'LOAD_LOCAL 0\nLOAD_LOCAL 1\n{op}\n', result),
                                             ('LOAD_LOCAL 0\n', left), ('LOAD_LOCAL 1\n', right)):
                    body += transport.FloatReconstruction.check(
                        expression + f'F64_TO_BITS\nPUSH_I64 {signed(expected)}\n', ordinal)
                    ordinal += 1
            _, c = self.paired('.entry main\n.function main 0 2 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')
            self.assertIn((ROOT / 'src/binary64_arithmetic.h').read_text(), c)
            self.inspect([op])

    def test_rounding_snapshots_calls_branches_and_loop(self):
        # (1+2^-27)*(1-2^-27) rounds to one before adding -1.
        text = '''.entry main
.function main 0 2 0 int 1
PUSH_F64 bits:3ff0000002000000
PUSH_F64 bits:3feffffffc000000
F64_MUL
PUSH_F64 bits:bff0000000000000
F64_ADD
F64_TO_BITS
PUSH_I64 0
I64_EQ
JMP_TRUE round_ok
PUSH_I64 1
RET
round_ok:
PUSH_F64 bits:4000000000000000
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
PUSH_F64 bits:3ff0000000000000
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
F64_SUB
F64_TO_BITS
PUSH_I64 4607182418800017408
I64_EQ
JMP_TRUE snapshot_ok
PUSH_I64 2
RET
snapshot_ok:
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 1
PUSH_I64 2
I64_LT_S
PUSH_F64 bits:3ff0000000000000
PUSH_F64 bits:3ff0000000000000
F64_ADD
PUSH_F64 bits:4000000000000000
F64_EQ
BOOL_AND
JMP_FALSE done
LOAD_LOCAL 1
PUSH_I64 0
I64_EQ
JMP_FALSE other
LOAD_LOCAL 0
PUSH_F64 bits:4000000000000000
F64_MUL
STORE_LOCAL 0
JMP joined
other:
LOAD_LOCAL 0
PUSH_F64 bits:4000000000000000
F64_DIV
STORE_LOCAL 0
joined:
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 0
F64_TO_BITS
PUSH_I64 4607182418800017408
I64_EQ
JMP_TRUE okay
PUSH_I64 3
RET
okay:
PUSH_I64 0
RET
.end
.function relay 1 1 0 float 1
.parameters relay float
LOAD_LOCAL 0
RET
.end
'''
        _, c = self.paired(text, 'shadow nlr_f1_relay { assert (== (float_to_bits (nlr_f1_relay (float_from_bits 1))) 1) }')
        self.inspect(list(FLOAT_ARITHMETIC))
        self.observe_call_count(c)
        # I also qualify the rounding boundary when the C compiler may contract.
        source = self.work / 'contract.c'
        source.write_text(c)
        import shlex
        self.command(shlex.split(os.environ.get('CC', 'cc')) +
                     ['-std=c11', '-O3', '-ffp-contract=fast', '-Wall', '-Wextra', '-Werror',
                      source, '-o', self.work / 'contract'])
        self.command([self.work / 'contract'])

    def test_exact_tag_and_missing_operand_refusals(self):
        for op in FLOAT_ARITHMETIC:
            module = {'functions': [{'params': [], 'result': INT, 'locals': 0,
                      'size': 1, 'code': [{'op': op, 'pc': 0, 'arg': 0}]}]}
            for tag in (INT, BOOL, 0, 7):
                for reverse in (False, True):
                    stack = [Expr(FLOAT, 'float_bits', 0), Expr(tag, 'constant', 0)]
                    with self.assertRaisesRegex(Refusal, 'exact scalar operand'):
                        Analyze(module, 0).simple(0, stack[::-1] if reverse else stack, set(), [])
            with self.assertRaisesRegex(Refusal, 'populated scalar'):
                Analyze(module, 0).simple(0, [Expr(FLOAT, 'float_bits', 0)], set(), [])

    test_refusals_preserve_outputs = transport.FloatReconstruction.test_float_operations_remain_refused_without_publication


if __name__ == '__main__':
    unittest.main()
