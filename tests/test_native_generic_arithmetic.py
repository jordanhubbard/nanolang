"""I retain the VM's bounded generic numeric contract in native C."""
from pathlib import Path
import tempfile
import unittest
from tests import test_native_floats as floats
ROOT = floats.ROOT


class NativeGenericArithmetic(unittest.TestCase):
    run_command = floats.NativeFloats.run_command
    checked = floats.NativeFloats.checked
    assemble = floats.NativeFloats.assemble
    native = floats.NativeFloats.native

    def paired(self, body, helpers=''):
        with tempfile.TemporaryDirectory(prefix='nano-generic-numeric-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, body, helpers)
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            native = self.checked([self.native(work, module, sanitize=True)])
            self.assertEqual(native.stdout, vm.stdout)

    def test_numeric_pair_matrix_and_exact_result_tags(self):
        body = ''
        for left_tag in ('I64', 'F64'):
            for right_tag in ('I64', 'F64'):
                floating = 'F64' in (left_tag, right_tag)
                for op, expected in [('ADD', 9), ('SUB', 5), ('MUL', 14),
                                     ('DIV', 3.5 if floating else 3)]:
                    tag, push, eq = (3, 'F64', 'F64_EQ') if floating else (1, 'I64', 'I64_EQ')
                    body += (f'PUSH_{left_tag} 7\nPUSH_{right_tag} 2\n{op}\n'
                             f'DUP\nTYPE_CHECK {tag}\nASSERT\nPUSH_{push} {expected}\n{eq}\nASSERT\n')
        self.paired(body)

    def test_float_rounding_zero_nan_and_infinity(self):
        body = ('PUSH_I64 9007199254740993\nPUSH_F64 0\nADD\n'
                'PUSH_F64 9007199254740992\nF64_EQ\nASSERT\n'
                'PUSH_F64 0\nNEG\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n')
        for lhs in ('PUSH_I64 -7', 'PUSH_F64 -7'):
            for rhs in ('PUSH_I64 0', 'PUSH_F64 0', 'PUSH_F64 -0'):
                body += lhs+'\n'+rhs+'\nDIV\nCAST_STRING\nPUSH_STR zero\nEQ\nASSERT\n'
        for op in ('ADD', 'SUB', 'MUL', 'DIV'):
            body += f'PUSH_F64 nan\nPUSH_I64 2\n{op}\nDUP\nF64_NE\nASSERT\n'
        body += 'PUSH_F64 inf\nNEG\nPUSH_F64 -inf\nF64_EQ\nASSERT\n'
        self.paired(body)

    def test_wrapped_integer_totality_and_checked_tagged_ints(self):
        cases = [(9223372036854775807, 1, 'ADD', -9223372036854775808),
                 (-9223372036854775808, 1, 'SUB', 9223372036854775807),
                 (9223372036854775807, 2, 'MUL', -2),
                 (-9223372036854775808, -1, 'DIV', -9223372036854775808),
                 (-9223372036854775808, -1, 'MOD', 0),
                 (7, 0, 'DIV', 0), (7, 0, 'MOD', 0), (-7, 2, 'MOD', -1)]
        body = ''
        for a,b,op,result in cases:
            for tagged in (False, True):
                operand = f'PUSH_I64 {a}\n'
                if tagged:
                    operand += 'STORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
                body += operand+f'PUSH_I64 {b}\n{op}\nPUSH_I64 {result}\nI64_EQ\nASSERT\n'
        body += 'PUSH_I64 -9223372036854775808\nNEG\nPUSH_I64 -9223372036854775808\nI64_EQ\nASSERT\n'
        self.paired(body)

    def test_calls_locals_and_eager_operand_effects(self):
        helpers = ('.function left 0 0 0 int 1\nPUSH_I64 11\nPRINTLN\nPUSH_I64 7\nRET\n.end\n'
                   '.function right 0 0 0 float 1\nPUSH_I64 22\nPRINTLN\nPUSH_F64 2\nRET\n.end\n'
                   '.function quotient 2 2 0 float 1\n.parameters quotient int float\n'
                   'LOAD_LOCAL 0\nLOAD_LOCAL 1\nDIV\nRET\n.end\n')
        self.paired('CALL left\nCALL right\nCALL quotient\nSTORE_LOCAL 0\n'
                    'LOAD_LOCAL 0\nNEG\nPUSH_F64 -3.5\nF64_EQ\nASSERT\n',helpers)

    def test_invalid_numeric_tags_and_float_mod_are_refused(self):
        for wrong in ('PUSH_BOOL 1', 'PUSH_U8 1', 'PUSH_VOID'):
            for op in ('ADD','SUB','MUL','DIV','MOD','NEG'):
                with self.subTest(wrong=wrong,op=op), tempfile.TemporaryDirectory(prefix='nano-generic-invalid-') as tmp:
                    work=Path(tmp)
                    body=wrong+'\n'+('' if op=='NEG' else 'PUSH_I64 2\n')+op+'\nPOP\n'
                    module=self.assemble(work,body)
                    self.assertNotEqual(self.run_command([ROOT/'bin/nano_vm',module]).returncode,0)
                    output=work/'input.c'; output.write_text('previous')
                    result=self.run_command([ROOT/'bin/nvm2c',module,'-o',output])
                    if result.returncode:
                        self.assertEqual(output.read_text(),'previous')
                    else:
                        self.assertNotEqual(self.run_command([self.native(work,module,sanitize=True)]).returncode,0)
        with tempfile.TemporaryDirectory(prefix='nano-float-mod-') as tmp:
            work=Path(tmp);module=self.assemble(work,'PUSH_F64 7\nPUSH_I64 2\nMOD\nPOP\n')
            self.assertNotEqual(self.run_command([ROOT/'bin/nano_vm',module]).returncode,0)
            output=work/'previous.c'; output.write_text('previous')
            self.assertNotEqual(self.run_command([ROOT/'bin/nvm2c',module,'-o',output]).returncode,0)
            self.assertEqual(output.read_text(),'previous')

    def test_unproved_mixed_promotion_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-unproved-') as tmp:
            work=Path(tmp)
            module=self.assemble(work,'PUSH_I64 2\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_F64 1.5\nADD\nPOP\n')
            self.checked([ROOT/'bin/nano_vm',module])
            output=work/'previous.c'; output.write_text('previous')
            result=self.run_command([ROOT/'bin/nvm2c',module,'-o',output])
            self.assertNotEqual(result.returncode,0)
            self.assertIn('known int or float',result.stderr)
            self.assertEqual(output.read_text(),'previous')


if __name__ == '__main__':
    unittest.main()
