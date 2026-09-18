"""I preserve typed division and remainder, including defined exceptional results."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


def quotient(a, b):
    if b == 0:
        return 0
    magnitude = abs(a) // abs(b)
    return addition.wrapped(-magnitude if (a < 0) != (b < 0) else magnitude)


class IntegerDivision(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_signed_endpoints_zero_and_nonexact_quotients(self):
        values = (LOW, LOW+1, -7, -3, -1, 0, 1, 3, 7, HIGH-1, HIGH)
        # Separate modules exercise needed-only helpers (no unused wrapping helper).
        for opcode, name in (('I64_DIV_S', 'div'), ('I64_REM_S', 'rem')):
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for a in values:
                    for b in values:
                        q = quotient(a, b)
                        result = q if name == 'div' else addition.wrapped(a - q*b) if b else 0
                        body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nCALL operation\nPUSH_I64 {result}\nI64_EQ\n'
                                 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                        '.function operation 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'+opcode+
                        '\nRET\n.end\n.parameters operation int int\n')
                expected = -2 if name == 'div' else -1
                shadows = ('shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                           f'shadow nlr_f1_operation {{ assert (== (nlr_f1_operation -7 3) {expected}) }}\n')
                self.paired(text, 0, shadows, names=('nlr_i64_'+name,))

    def test_division_pretest_loop_and_remainder_snapshot(self):
        text = '''.entry main
.function main 0 2 0 int 1
PUSH_I64 81
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 3
I64_DIV_S
PUSH_I64 0
I64_GT_S
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 3
I64_DIV_S
STORE_LOCAL 0
JMP loop
done:
PUSH_I64 -7
STORE_LOCAL 1
LOAD_LOCAL 1
PUSH_I64 99
STORE_LOCAL 1
PUSH_I64 3
I64_REM_S
PUSH_I64 -1
I64_EQ
JMP_FALSE bad
LOAD_LOCAL 0
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

    def test_wrong_tag_refusal_preserves_previous_output(self):
        for opcode in ('I64_DIV_S', 'I64_REM_S'):
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
