"""I preserve both carry/borrow results and exact input-bit normalization."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


class CarryBorrow(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_endpoints_and_noncanonical_carry_bits(self):
        values = (LOW, LOW + 1, -1, 0, 1, HIGH - 1, HIGH)
        carries = (LOW, -3, -2, -1, 0, 1, 2, 3, HIGH)
        for opcode in ('I64_ADD_CARRY', 'I64_SUB_BORROW'):
            for group in (carries[:3], carries[3:6], carries[6:]):
                with self.subTest(opcode=opcode, carries=group):
                    body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                    for left in values:
                        for right in values:
                            for carry in group:
                                unsigned_left = left % (1 << 64)
                                unsigned_right = right % (1 << 64)
                                bit = carry & 1
                                exact = (unsigned_left + unsigned_right + bit if opcode == 'I64_ADD_CARRY'
                                         else unsigned_left - unsigned_right - bit)
                                low = addition.wrapped(exact)
                                high = int(exact >= 1 << 64 if opcode == 'I64_ADD_CARRY' else exact < 0)
                                body += (f'PUSH_I64 {left}\nPUSH_I64 {right}\nPUSH_I64 {carry}\n{opcode}\n'
                                         f'STORE_LOCAL 1\nPUSH_I64 {low}\nI64_EQ\nLOAD_LOCAL 0\n'
                                         f'BOOL_AND\nSTORE_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 {high}\n'
                                         'I64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                    body += ('LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\n'
                             'bad:\nPUSH_I64 1\nRET\n')
                    self.paired('.entry main\n.function main 0 2 0 int 1\n' + body + '.end\n', 0,
                                'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_loop_high_result_and_call_snapshot(self):
        text = '''.entry main
.function main 0 3 0 int 1
PUSH_I64 0
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 -3
PUSH_I64 0
I64_ADD_CARRY
SWAP
POP
PUSH_I64 0
I64_EQ
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 3
I64_EQ
JMP_FALSE bad
PUSH_I64 -1
STORE_LOCAL 0
LOAD_LOCAL 0
CALL identity
PUSH_I64 0
PUSH_I64 1
I64_ADD_CARRY
PUSH_I64 99
STORE_LOCAL 0
STORE_LOCAL 1
STORE_LOCAL 2
LOAD_LOCAL 1
PUSH_I64 1
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 2
PUSH_I64 0
I64_EQ
JMP_FALSE bad
PUSH_I64 0
PUSH_I64 -1
PUSH_I64 1
I64_SUB_BORROW
PUSH_I64 1
I64_EQ
SWAP
PUSH_I64 0
I64_EQ
BOOL_AND
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function identity 1 1 0 int 1
LOAD_LOCAL 0
RET
.end
.parameters identity int
'''
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                            'shadow nlr_f1_identity { assert (== (nlr_f1_identity -1) -1) }')

    def test_tag_and_arity_refusals_preserve_output(self):
        for opcode in ('I64_ADD_CARRY', 'I64_SUB_BORROW'):
            for inputs in ('PUSH_I64 0\nPUSH_I64 0',
                           'PUSH_I64 0\nPUSH_I64 0\nPUSH_BOOL 1'):
                with self.subTest(opcode=opcode, inputs=inputs), \
                     tempfile.TemporaryDirectory(prefix='nano-hl-carry-refusal-') as tmp:
                    directory = Path(tmp)
                    source = directory / 'input.nasm'
                    output = directory / 'previous.nvm'
                    source.write_text('.entry main\n.function main 0 0 0 int 1\n' + inputs +
                                      '\n' + opcode + '\nPOP\nRET\n.end\n')
                    output.write_bytes(b'previous')
                    result = subprocess.run([ROOT / 'bin/nanoisa', 'asm', source, '-o', output],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertTrue('underflow' in result.stderr or
                                    'expects int but the operand is bool' in result.stderr,
                                    result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__':
    unittest.main()
