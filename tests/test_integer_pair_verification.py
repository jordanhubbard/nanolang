"""I execute exact integer result pairs through VM and standalone native C."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1


def signed(value):
    value &= MASK
    return value if value < (1 << 63) else value - (1 << 64)


class IntegerPairVerification(unittest.TestCase):
    def checked(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def paired(self, body, functions=''):
        with tempfile.TemporaryDirectory(prefix='integer-pairs-') as temporary:
            work = Path(temporary)
            assembly, module = work/'valid.nasm', work/'valid.nvm'
            assembly.write_text('.entry main\n.function main 0 2 0 int 1\n' + body +
                                'PUSH_I64 0\nRET\n.end\n' + functions)
            self.checked(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
            self.checked(ROOT/'bin/nano_vm', '--verify-only', module)
            self.checked(ROOT/'bin/nano_vm', module)
            source, binary = work/'valid.c', work/'valid'
            self.checked(ROOT/'bin/nvm2c', module, '-o', source)
            self.assertNotIn('__int128', source.read_text())
            self.checked(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=address,undefined', '-fno-omit-frame-pointer', source, '-lm', '-o', binary)
            self.checked(binary)

    def test_carry_borrow_low_bit_and_both_results(self):
        body = ''
        for opcode in ('I64_ADD_CARRY', 'I64_SUB_BORROW'):
            for a, b, carry in ((0, 0, 0), (7, 3, 1), (-1, 0, 1), (0, 0, 1),
                                (-1, -1, 3), (0, -1, 2), (-(1 << 63), 1, -1)):
                total = (a & MASK) + (b & MASK) + (carry & 1) if opcode == 'I64_ADD_CARRY' else (a & MASK) - (b & MASK) - (carry & 1)
                high = int(total > MASK) if opcode == 'I64_ADD_CARRY' else int(total < 0)
                body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nPUSH_I64 {carry}\n{opcode}\n'
                         f'PUSH_I64 {high}\nI64_EQ\nASSERT\n'
                         f'PUSH_I64 {signed(total)}\nI64_EQ\nASSERT\n')
        self.paired(body)

    def test_signed_unsigned_high_words_and_composition(self):
        body = ''
        for opcode in ('I64_MUL_WIDE_S', 'I64_MUL_WIDE_U'):
            for a, b in ((0, 7), (-1, 2), (-1, -1), (-(1 << 63), 2), ((1 << 63)-1, (1 << 63)-1)):
                product = a*b if opcode.endswith('_S') else (a & MASK)*(b & MASK)
                body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\n{opcode}\n'
                         f'PUSH_I64 {signed(product >> 64)}\nI64_EQ\nASSERT\n'
                         f'PUSH_I64 {signed(product)}\nI64_EQ\nASSERT\n')
        # I consume both pair results directly as the next arithmetic operands.
        body += 'PUSH_I64 -1\nPUSH_I64 1\nPUSH_I64 0\nI64_ADD_CARRY\nPUSH_I64 0\nI64_SUB_BORROW\nPUSH_I64 1\nI64_EQ\nASSERT\nPUSH_I64 -1\nI64_EQ\nASSERT\n'
        self.paired(body)

    def test_integer_pairs_through_loop_locals(self):
        self.paired('PUSH_I64 0\nSTORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 1\n'
                    'loop:\nLOAD_LOCAL 0\nPUSH_I64 8\nI64_LT_S\nJMP_FALSE done\n'
                    'LOAD_LOCAL 1\nPUSH_I64 7\nPUSH_I64 1\nI64_ADD_CARRY\n'
                    'PUSH_I64 0\nI64_EQ\nASSERT\nSTORE_LOCAL 1\n'
                    'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\n'
                    'done:\nLOAD_LOCAL 1\nPUSH_I64 64\nI64_EQ\nASSERT\n')

    def test_wide_product_bit_patterns(self):
        values = (0, 1, -1, 1 << 31, 1 << 32, -(1 << 32), (1 << 63)-1,
                  -(1 << 63), signed(0xaaaaaaaaaaaaaaaa), signed(0x5555555555555555))
        for opcode in ('I64_MUL_WIDE_S', 'I64_MUL_WIDE_U'):
            body = ''
            for a in values:
                for b in values:
                    product = a*b if opcode.endswith('_S') else (a & MASK)*(b & MASK)
                    body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\n{opcode}\n'
                             f'PUSH_I64 {signed(product >> 64)}\nI64_EQ\nASSERT\n'
                             f'PUSH_I64 {signed(product)}\nI64_EQ\nASSERT\n')
            self.paired(body)

    def test_calls_and_reaching_pair_joins(self):
        self.paired('PUSH_I64 -1\nPUSH_I64 0\nPUSH_I64 1\nCALL low\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                    'PUSH_I64 -1\nPUSH_I64 2\nCALL high\nPUSH_I64 -1\nI64_EQ\nASSERT\n'
                    'PUSH_I64 0\nSTORE_LOCAL 0\nagain:\nLOAD_LOCAL 0\nPUSH_I64 2\nI64_LT_S\nJMP_FALSE done\n'
                    'LOAD_LOCAL 0\nPUSH_I64 0\nI64_EQ\nJMP_FALSE second\n'
                    'PUSH_I64 -1\nPUSH_I64 1\nPUSH_I64 0\nI64_ADD_CARRY\nJMP joined\n'
                    'second:\nPUSH_I64 0\nPUSH_I64 -1\nPUSH_I64 1\nI64_SUB_BORROW\n'
                    'joined:\nPUSH_I64 1\nI64_EQ\nASSERT\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                    'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP again\ndone:\n',
                    '.function low 3 3 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nI64_ADD_CARRY\nPOP\nRET\n.end\n'
                    '.parameters low int int int\n'
                    '.function high 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nI64_MUL_WIDE_S\nSWAP\nPOP\nRET\n.end\n'
                    '.parameters high int int\n')
