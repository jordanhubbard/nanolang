"""I retain unsigned integer ordering and boolean result identity."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition
from tests.test_reconstructed_integer_bitwise import VALUES

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


class UnsignedComparisons(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_unsigned_endpoint_order_and_bool_returns(self):
        for opcode, name, operation in (('I64_LT_U', 'lt_u', lambda a,b: a < b),
                                        ('I64_LE_U', 'le_u', lambda a,b: a <= b),
                                        ('I64_GT_U', 'gt_u', lambda a,b: a > b),
                                        ('I64_GE_U', 'ge_u', lambda a,b: a >= b)):
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for a in VALUES:
                    for b in VALUES:
                        expected = operation(a % (1 << 64), b % (1 << 64))
                        body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nCALL compare\n'+
                                 ('' if expected else 'BOOL_NOT\n')+
                                 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                        '.function compare 2 2 0 bool 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'+opcode+
                        '\nRET\n.end\n.parameters compare int int\n')
                expected = 'true' if operation(0, (1 << 64)-1) else 'false'
                shadows = ('shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                           f'shadow nlr_f1_compare {{ assert (== (nlr_f1_compare 0 -1) {expected}) }}\n')
                self.paired(text, 0, shadows, names=('nlr_i64_'+name,))

    def test_unsigned_loop_crossing_sign_boundary_and_snapshot(self):
        text = f'''.entry main
.function main 0 2 0 int 1
PUSH_I64 {HIGH}
STORE_LOCAL 0
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 {LOW+2}
I64_LT_U
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 0
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 {HIGH}
I64_GT_U
LOAD_LOCAL 1
PUSH_I64 3
I64_EQ
BOOL_AND
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
'''
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_wrong_input_and_result_tags_preserve_output(self):
        for opcode in ('I64_LT_U', 'I64_LE_U', 'I64_GT_U', 'I64_GE_U'):
            for wrong_input in (True, False):
                with self.subTest(opcode=opcode, wrong_input=wrong_input), tempfile.TemporaryDirectory(prefix='nano-hl-ucompare-refusal-') as tmp:
                    directory = Path(tmp); source = directory/'input.nasm'; output = directory/'previous.nvm'
                    rhs = 'PUSH_BOOL 0' if wrong_input else 'PUSH_I64 0'
                    # Int return deliberately rejects the comparison's bool result.
                    source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                      'PUSH_I64 7\n'+rhs+'\n'+opcode+'\nRET\n.end\n')
                    output.write_bytes(b'previous')
                    result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                    self.assertIn('expects int but the operand is bool', result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__': unittest.main()
