"""I compare total integer reconstruction from one retained module."""
from pathlib import Path
import subprocess
import os
from unittest import mock
import tempfile
import unittest
from tests import test_scalar_reconstruction as base

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = -(1 << 63), (1 << 63) - 1


def wrapped(value):
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


class IntegerReconstruction(unittest.TestCase):
    def checked(self, args, expected=0):
        if Path(args[0]).name == 'nanoc_c':
            flags = os.environ.get('NANO_CFLAGS', '') + ' -fsanitize=undefined -fno-sanitize-recover=all'
            with mock.patch.dict(os.environ, {'NANO_CFLAGS': flags}):
                return base.ScalarReconstruction.checked(self, args, expected)
        return base.ScalarReconstruction.checked(self, args, expected)

    paired = base.ScalarReconstruction.paired
    def assemble(self, directory, text):
        module = base.ScalarReconstruction.assemble(self, directory, text)
        dumped = self.checked([ROOT/'bin/nanoisa', 'dump', module]).stdout
        assembly = directory/'roundtrip.nasm'; assembly.write_text(dumped)
        rebuilt = directory/'roundtrip.nvm'
        self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
        self.assertEqual(module.read_bytes(), rebuilt.read_bytes())
        assembly.unlink()
        return module

    def test_signed_endpoints_and_cancellation(self):
        values = (LOW, LOW+1, -1, 0, 1, HIGH-1, HIGH)
        body = 'PUSH_BOOL 1\nSTORE_LOCAL 0\n'
        for op, name, operation in (('I64_ADD','add',lambda a,b:a+b),
                                    ('I64_SUB','sub',lambda a,b:a-b)):
            for a in values:
                for b in values:
                    body += (f'PUSH_I64 {a}\nPUSH_I64 {b}\nCALL {name}\n'
                             f'PUSH_I64 {wrapped(operation(a,b))}\nI64_EQ\n'
                             'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
        for a in values:
            body += (f'PUSH_I64 {a}\nCALL neg\nPUSH_I64 {wrapped(-a)}\nI64_EQ\n'
                     'LOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
        # I snapshot a local before a later store changes its value.
        body += (f'PUSH_I64 {HIGH}\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 1\nSTORE_LOCAL 1\n'
                 f'PUSH_I64 1\nI64_ADD\nPUSH_I64 {LOW}\nI64_EQ\nLOAD_LOCAL 0\nBOOL_AND\nSTORE_LOCAL 0\n')
        body += 'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n'
        helpers = ''
        for op, name, arity in (('I64_ADD','add',2),('I64_SUB','sub',2),('I64_NEG','neg',1)):
            helpers += (f'.function {name} {arity} {arity} 0 int 1\nLOAD_LOCAL 0\n' +
                        ('LOAD_LOCAL 1\n' if arity == 2 else '') + op + '\nRET\n.end\n' +
                        f'.parameters {name}' + ' int'*arity + '\n')
        shadows = '''shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }
shadow nlr_f1_add { assert (== (nlr_f1_add 4 5) 9) }
shadow nlr_f2_sub { assert (== (nlr_f2_sub 4 5) -1) }
shadow nlr_f3_neg { assert (== (nlr_f3_neg -5) 5) }
'''
        self.paired('.entry main\n.function main 0 2 0 int 1\n'+body+'.end\n'+helpers,0,shadows)

    def test_arithmetic_pretest_loop(self):
        text = f'''.entry main
.function main 0 2 0 int 1
PUSH_I64 0
STORE_LOCAL 0
PUSH_I64 {HIGH-1}
STORE_LOCAL 1
loop:
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
PUSH_I64 4
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 1
PUSH_I64 {LOW+1}
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

    def test_remaining_arithmetic_and_type_refusals_preserve_output(self):
        cases = ('PUSH_I64 1\nPUSH_I64 2\nI64_DIV_U',
                 'PUSH_I64 1\nPUSH_I64 2\nADD',
                 'PUSH_BOOL 1\nI64_NEG',
                 'PUSH_BOOL 1\nPUSH_I64 2\nI64_ADD')
        for body in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix='nano-hl-add-refusal-') as tmp:
                directory=Path(tmp)
                text='.entry main\n.function main 0 0 0 int 1\n'+body+'\nRET\n.end\n'
                if 'PUSH_BOOL' in body:
                    assembly=directory/'input.nasm'; assembly.write_text(text)
                    output=directory/'previous.nvm'; output.write_bytes(b'previous')
                    result=subprocess.run([ROOT/'bin/nanoisa','asm',assembly,'-o',output],capture_output=True,text=True)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertIn('expects int but the operand is bool',result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous')
                    continue
                module=self.assemble(directory,text)
                for language in ('c','nano'):
                    output=directory/'previous'; output.write_bytes(b'previous')
                    result=subprocess.run([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],capture_output=True,text=True)
                    self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                    self.assertIn('I did not publish reconstructed source',result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous')


if __name__ == '__main__': unittest.main()
