"""I reconstruct exact wrapped multiplication without signed overflow."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = addition.LOW, addition.HIGH


class IntegerMultiplication(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    paired = addition.IntegerReconstruction.paired

    def test_signed_digits_endpoints_and_high_products(self):
        values = (LOW, LOW+1, -(1 << 32), -3, -1, 0, 1, 3, 1 << 32, HIGH-1, HIGH)
        body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
        for left in values:
            for right in values:
                result = addition.wrapped(left * right)
                body += (f'PUSH_I64 {left}\nPUSH_I64 {right}\nCALL product\nPUSH_I64 {result}\nI64_EQ\n'
                         'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
        body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
        text = ('.entry main\n.function main 0 1 0 int 1\n'+body+'.end\n'
                '.function product 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nI64_MUL\nRET\n.end\n'
                '.parameters product int int\n')
        shadows = '''shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }
shadow nlr_f1_product { assert (== (nlr_f1_product -3 7) -21) }
'''
        self.paired(text,0,shadows,names=('nlr_i64_mul',))

    def test_multiplication_in_structured_loop(self):
        text = f'''.entry main
.function main 0 2 0 int 1
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 {HIGH}
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 3
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 1
PUSH_I64 -3
I64_MUL
STORE_LOCAL 1
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 {addition.wrapped(HIGH * (-3)**3)}
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
'''
        self.paired(text,0,'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_bool_multiplication_refused_before_publication(self):
        with tempfile.TemporaryDirectory(prefix='nano-hl-mul-refusal-') as tmp:
            directory=Path(tmp);source=directory/'input.nasm';output=directory/'previous.nvm'
            source.write_text('.entry main\n.function main 0 0 0 int 1\n'
                              'PUSH_I64 7\nPUSH_BOOL 0\nI64_MUL\nRET\n.end\n')
            output.write_bytes(b'previous')
            result=subprocess.run([ROOT/'bin/nanoisa','asm',source,'-o',output],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            self.assertIn('expects int but the operand is bool',result.stderr)
            self.assertEqual(output.read_bytes(),b'previous')

if __name__ == '__main__': unittest.main()
