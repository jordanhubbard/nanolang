"""I preserve exact64-bit patterns across structured reconstruction surfaces."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH
VALUES = (LOW, LOW+1, -7, -1, 0, 1, 7, HIGH-1, HIGH,
          0x5555555555555555, addition.wrapped(0xAAAAAAAAAAAAAAAA))


class IntegerBitwise(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_binary_patterns_and_signed_endpoints(self):
        for opcode, name, operation in (('I64_AND', 'band', lambda a,b: a & b),
                                        ('I64_OR', 'bor', lambda a,b: a | b),
                                        ('I64_XOR', 'bxor', lambda a,b: a ^ b)):
            with self.subTest(opcode=opcode):
                body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
                for a in VALUES:
                    for b in VALUES:
                        result = addition.wrapped(operation(a, b))
                        body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nCALL operation\nPUSH_I64 {result}\nI64_EQ\n'
                                 'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
                body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
                text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                        '.function operation 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n'+opcode+
                        '\nRET\n.end\n.parameters operation int int\n')
                expected = addition.wrapped(operation(-7, 3))
                shadows = ('shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                           f'shadow nlr_f1_operation {{ assert (== (nlr_f1_operation -7 3) {expected}) }}\n')
                self.paired(text, 0, shadows, names=('nlr_i64_'+name,))

    def test_inversion_only_and_double_inversion(self):
        body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
        for value in VALUES:
            for repeat in (1, 2):
                expected = addition.wrapped(~value) if repeat == 1 else value
                body += (f'PUSH_I64 {value}\n'+'CALL invert\n'*repeat+f'PUSH_I64 {expected}\nI64_EQ\n'
                         'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
        body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
        text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                '.function invert 1 1 0 int 1\nLOAD_LOCAL 0\nI64_INVERT\nRET\n.end\n.parameters invert int\n')
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\n'
                    'shadow nlr_f1_invert { assert (== (nlr_f1_invert 0) -1) }', names=('nlr_i64_invert',))

    def test_loop_and_inversion_snapshot(self):
        text = '''.entry main
.function main 0 2 0 int 1
PUSH_I64 1
STORE_LOCAL 0
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 16
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 1
LOAD_LOCAL 0
I64_OR
STORE_LOCAL 1
LOAD_LOCAL 0
PUSH_I64 1
I64_SHL
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 0
STORE_LOCAL 1
I64_INVERT
PUSH_I64 -16
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
        for opcode in ('I64_AND', 'I64_OR', 'I64_XOR', 'I64_INVERT'):
            with self.subTest(opcode=opcode), tempfile.TemporaryDirectory(prefix='nano-hl-bitwise-refusal-') as tmp:
                directory = Path(tmp); source = directory/'input.nasm'; output = directory/'previous.nvm'
                operands = '' if opcode == 'I64_INVERT' else 'PUSH_I64 7\n'
                source.write_text('.entry main\n.function main 0 0 0 int 1\n'+operands+
                                  'PUSH_BOOL 0\n'+opcode+'\nRET\n.end\n')
                output.write_bytes(b'previous')
                result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn('expects int but the operand is bool', result.stderr)
                self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__': unittest.main()
