"""I coerce enum ordinals only for the VM's typed binary I64 operations."""
from pathlib import Path
import signal
import tempfile
import unittest
from tests import test_native_enum_scalars as enums
ROOT=enums.ROOT


class TypedEnum(unittest.TestCase):
    run_command=enums.EnumScalars.run_command
    checked=enums.EnumScalars.checked
    assemble=enums.EnumScalars.assemble
    native=enums.EnumScalars.native
    paired=enums.EnumScalars.paired

    def test_all_typed_binary_operations_and_orders(self):
        body=''
        operations=[('I64_ADD',9),('I64_SUB',5),('I64_MUL',14),('I64_DIV_S',3),
                    ('I64_REM_S',1),('I64_EQ',False),('I64_NE',True),('I64_LT_S',False),
                    ('I64_LE_S',False),('I64_GT_S',True),('I64_GE_S',True)]
        for left,right in [('ENUM_VAL 0 7','ENUM_VAL 0 2'),('ENUM_VAL 0 7','PUSH_I64 2'),
                           ('PUSH_I64 7','ENUM_VAL 0 2')]:
            for op,value in operations:
                boolean=isinstance(value,bool)
                body+=f'{left}\n{right}\n{op}\nDUP\nTYPE_CHECK {4 if boolean else 1}\nASSERT\n'
                body+=f'PUSH_{"BOOL" if boolean else "I64"} {int(value)}\nEQ\nASSERT\n'
        self.paired(body)

    def test_stored_and_duplicated_enum_tags_survive_consumption(self):
        helpers=('.function plus 2 2 0 int 1\n.parameters plus int int\n'
                 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nI64_ADD\nRET\n.end\n')
        body=('ENUM_VAL 0 5\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nDUP\nPUSH_I64 1\n'
              'I64_ADD\nPUSH_I64 6\nEQ\nASSERT\nTYPE_CHECK 9\nASSERT\n'
              'LOAD_LOCAL 0\nPUSH_I64 2\nCALL plus\nPUSH_I64 7\nEQ\nASSERT\n'
              'PUSH_I64 2\nLOAD_LOCAL 0\nCALL plus\nPUSH_I64 7\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nTYPE_CHECK 9\nASSERT\n')
        self.paired(body,helpers)

    def test_wrapping_signed_division_and_zero_totality(self):
        body=''
        for left,right,op,value in [
            ('PUSH_I64 9223372036854775807','ENUM_VAL 0 1','I64_ADD',-9223372036854775808),
            ('PUSH_I64 -9223372036854775808','ENUM_VAL 0 1','I64_SUB',9223372036854775807),
            ('PUSH_I64 9223372036854775807','ENUM_VAL 0 2','I64_MUL',-2),
            ('PUSH_I64 -7','ENUM_VAL 0 2','I64_DIV_S',-3),
            ('PUSH_I64 -7','ENUM_VAL 0 2','I64_REM_S',-1),
            ('ENUM_VAL 0 7','ENUM_VAL 0 0','I64_DIV_S',0),
            ('ENUM_VAL 0 7','ENUM_VAL 0 0','I64_REM_S',0)]:
            body+=f'{left}\n{right}\n{op}\nPUSH_I64 {value}\nEQ\nASSERT\n'
        self.paired(body)

    def refused(self, body):
        with tempfile.TemporaryDirectory(prefix='nano-typed-enum-refusal-') as tmp:
            work=Path(tmp);module=self.assemble(work,body+'POP\n')
            vm=self.run_command([ROOT/'bin/nano_vm',module])
            self.assertNotEqual(vm.returncode,0)
            native=self.run_command([self.native(work,module,sanitize=True)])
            self.assertEqual(native.returncode,-signal.SIGABRT,native.stderr)
            self.assertIn('I stopped at a native invariant',native.stderr)
            for error in ('Sanitizer','runtime error:'):
                self.assertNotIn(error,native.stderr)

    def test_unrelated_tags_are_not_integer_operands(self):
        for wrong in ('PUSH_BOOL 1','PUSH_U8 1','PUSH_F64 1','PUSH_VOID','PUSH_STR zero'):
            for op in ('I64_ADD','I64_REM_S','I64_LT_S'):
                with self.subTest(wrong=wrong,op=op):
                    self.refused(wrong+'\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nENUM_VAL 0 2\n'+op+'\n')

    def test_unary_and_generic_integer_guards_remain_exact(self):
        for body in ('ENUM_VAL 0 2\nI64_NEG\n','ENUM_VAL 0 2\nNEG\n',
                     'ENUM_VAL 0 2\nPUSH_I64 1\nMOD\n',
                     'ENUM_VAL 0 2\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_F64 1\nF64_ADD\n'):
            with self.subTest(body=body):
                self.refused(body)

    def test_known_float_mismatch_preserves_previous_module(self):
        with tempfile.TemporaryDirectory(prefix='nano-enum-float-profile-') as tmp:
            work=Path(tmp);source=work/'invalid.nasm';output=work/'previous.nvm'
            source.write_text('.types 0 1 0\n.entry main\n.function main 0 0 0 int 1\n'
                              'ENUM_VAL 0 2\nPUSH_F64 1\nF64_ADD\nPOP\nPUSH_I64 0\nRET\n.end\n')
            output.write_bytes(b'previous')
            result=self.run_command([ROOT/'bin/nanoisa','asm',source,'-o',output])
            self.assertNotEqual(result.returncode,0)
            self.assertIn('expects float',result.stderr)
            self.assertEqual(output.read_bytes(),b'previous')
