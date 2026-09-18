"""I preserve unsigned division and remainder with exact signed carriers."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


def quotient(a, b):
    return addition.wrapped((a % (1 << 64)) // (b % (1 << 64))) if b else 0


class UnsignedDivision(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_unsigned_endpoints_zero_and_high_bit_quotients(self):
        values = (LOW, LOW+1, LOW+17, -7, -3, -1, 0, 1, 3, 7, (1 << 32), HIGH-1, HIGH)
        # Separate modules exercise needed-only helpers (no unused wrapping helper).
        for opcode, name in (('I64_DIV_U', 'div_u'), ('I64_REM_U', 'rem_u')):
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for a in values:
                    for b in values:
                        q = quotient(a, b)
                        result = q if name == 'div_u' else addition.wrapped((a % (1 << 64)) % (b % (1 << 64))) if b else 0
                        body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nCALL operation\nPUSH_I64 {result}\nI64_EQ\n'
                                 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                        '.function operation 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'+opcode+
                        '\nRET\n.end\n.parameters operation int int\n')
                expected = 6148914691236517203 if name == 'div_u' else 0
                shadows = ('shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                           f'shadow nlr_f1_operation {{ assert (== (nlr_f1_operation -7 3) {expected}) }}\n')
                self.paired(text, 0, shadows, names=('nlr_i64_'+name,))

    def test_unsigned_division_loop_and_remainder_snapshot(self):
        text = f''' .entry main
.function main 0 2 0 int 1
PUSH_I64 -1
STORE_LOCAL 0
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 0
I64_NE
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 2
I64_DIV_U
STORE_LOCAL 0
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 64
I64_EQ
JMP_FALSE bad
PUSH_I64 -1
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 {HIGH}
I64_REM_U
PUSH_I64 1
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
'''
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_missing_operand_refusal_preserves_previous_output(self):
        for opcode in ('I64_DIV_U', 'I64_REM_U'):
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory(prefix='nano-hl-udiv-arity-') as tmp:
                directory = Path(tmp); source = directory/'input.nasm'; output = directory/'previous.nvm'
                source.write_text('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 7\n'+opcode+'\nRET\n.end\n')
                output.write_bytes(b'previous')
                result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn('underflow', result.stderr)
                self.assertEqual(output.read_bytes(), b'previous')

    def test_wrong_tag_refusal_preserves_previous_output(self):
        for opcode in ('I64_DIV_U', 'I64_REM_U'):
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory(prefix='nano-hl-div-refusal-') as tmp:
                directory = Path(tmp); source = directory/'input.nasm'; output = directory/'previous.nvm'
                source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                  'PUSH_I64 7\nPUSH_BOOL 0\n'+opcode+'\nRET\n.end\n')
                output.write_bytes(b'previous')
                result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn('expects int but the operand is bool', result.stderr)
                self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__': unittest.main()
