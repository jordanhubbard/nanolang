"""I compare masked shifts without relying on signed host shift behavior."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


class IntegerShifts(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_all_shift_families_and_masked_counts(self):
        values = (LOW, LOW+1, -7, -1, 0, 1, 7, HIGH-1, HIGH)
        counts = (LOW, LOW+1, -65, -64, -63, -1, 0, 1, 2, 63, 64, 65, HIGH)
        operations = (('I64_SHL', 'shl', lambda a,b: addition.wrapped(a << b)),
                      ('I64_SHR_S', 'shr_s', lambda a,b: a >> b),
                      ('I64_SHR_U', 'shr_u', lambda a,b: addition.wrapped((a % (1 << 64)) >> b)))
        for opcode, name, operation in operations:
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for value in values:
                    for count in counts:
                        result = operation(value, count & 63)
                        body += (f'PUSH_I64 {value}\nPUSH_I64 {count}\nCALL operation\nPUSH_I64 {result}\nI64_EQ\n'
                                 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                        '.function operation 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'+opcode+
                        '\nRET\n.end\n.parameters operation int int\n')
                expected = operation(-7, 1)
                shadows = ('shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                           f'shadow nlr_f1_operation {{ assert (== (nlr_f1_operation -7 1) {expected}) }}\n')
                self.paired(text, 0, shadows, names=('nlr_i64_'+name,))

    def test_shift_loop_condition_and_snapshots(self):
        text = '''.entry main
.function main 0 2 0 int 1
PUSH_I64 16
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
PUSH_I64 1
I64_SHR_U
PUSH_I64 0
I64_GT_S
JMP_FALSE done
LOAD_LOCAL 0
PUSH_I64 1
I64_SHR_S
STORE_LOCAL 0
JMP loop
done:
PUSH_I64 1
STORE_LOCAL 1
LOAD_LOCAL 0
LOAD_LOCAL 1
PUSH_I64 8
STORE_LOCAL 1
I64_SHL
PUSH_I64 2
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
        for opcode in ('I64_SHL', 'I64_SHR_S', 'I64_SHR_U'):
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory(prefix='nano-hl-shift-refusal-') as tmp:
                directory = Path(tmp); source = directory/'input.nasm'; output = directory/'previous.nvm'
                source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                                  'PUSH_I64 7\nPUSH_BOOL 0\n'+opcode+'\nRET\n.end\n')
                output.write_bytes(b'previous')
                result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn('expects int but the operand is bool', result.stderr)
                self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__': unittest.main()
