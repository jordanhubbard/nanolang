"""I preserve raw enum scalar tags and the VM's bounded arithmetic contract."""
from pathlib import Path
import signal
import tempfile
import unittest
from tests import test_native_tagged_arithmetic as tagged
ROOT = tagged.ROOT


class EnumScalars(unittest.TestCase):
    run_command = tagged.TaggedArithmetic.run_command
    checked = tagged.TaggedArithmetic.checked
    native = tagged.TaggedArithmetic.native
    paired = tagged.TaggedArithmetic.paired

    def assemble(self, work, body, helpers=''):
        return tagged.TaggedArithmetic.assemble(self, work, body, '.types 0 1 0\n'+helpers)

    def test_arithmetic_matrix_and_tags(self):
        body=''
        for left,right in [('ENUM_VAL 0 7','ENUM_VAL 0 2'),
                           ('ENUM_VAL 0 7','PUSH_I64 2'),
                           ('PUSH_I64 7','ENUM_VAL 0 2'),
                           ('ENUM_VAL 0 7','PUSH_F64 2'),
                           ('PUSH_F64 7','ENUM_VAL 0 2')]:
            floating='F64' in left+right
            for op,value in [('ADD',9),('SUB',5),('MUL',14),('DIV',3.5 if floating else 3)]:
                body+=f'{left}\n{right}\n{op}\nDUP\nTYPE_CHECK {3 if floating else 1}\nASSERT\n'
                body+=f'PUSH_{"F64" if floating else "I64"} {value}\nEQ\nASSERT\n'
        self.paired(body)

    def test_local_global_call_return_and_join_tags(self):
        helpers=('.function identity 1 1 0 enum 1\n.parameters identity enum\n'
                 'LOAD_LOCAL 0\nRET\n.end\n'
                 '.function tail 1 1 0 enum 1\n.parameters tail enum\nLOAD_LOCAL 0\nTAIL_CALL identity\n.end\n')
        body='ENUM_VAL 0 65535\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL tail\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\n'
        body+='DUP\nTYPE_CHECK 9\nASSERT\nCAST_INT\nPUSH_I64 65535\nEQ\nASSERT\n'
        for flag in (0,1):
            body+=f'PUSH_BOOL {flag}\nJMP_FALSE other{flag}\nENUM_VAL 0 2\nJMP joined{flag}\nother{flag}:\nENUM_VAL 0 3\njoined{flag}:\nDUP\nTYPE_CHECK 9\nASSERT\nCAST_INT\nPUSH_I64 {2 if flag else 3}\nEQ\nASSERT\n'
        self.paired(body,helpers)

    def test_casts_truthiness_and_comparison_compatibility(self):
        body=''
        for ordinal in (0,1,65535):
            body+=f'ENUM_VAL 0 {ordinal}\nCAST_BOOL\nPUSH_BOOL {int(bool(ordinal))}\nEQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nCAST_FLOAT\nPUSH_F64 0\nF64_EQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nCAST_STRING\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nPUSH_I64 {ordinal}\nEQ\nASSERT\n'
            body+=f'ENUM_VAL 0 {ordinal}\nENUM_VAL 0 {ordinal}\nEQ\nASSERT\n'
        body+='ENUM_VAL 0 0\nPRINTLN\nENUM_VAL 0 65535\nPRINTLN\n'
        body+='ENUM_VAL 0 2\nENUM_VAL 0 3\nNE\nASSERT\n'
        body+='ENUM_VAL 0 2\nENUM_VAL 0 3\nLE\nASSERT\n'
        body+='ENUM_VAL 0 2\nENUM_VAL 0 3\nGE\nASSERT\n'
        body+='ENUM_VAL 0 2\nPUSH_I64 3\nLT\nASSERT\n'
        body+='PUSH_I64 3\nENUM_VAL 0 2\nGT\nASSERT\n'
        body+='ENUM_VAL 0 2\nPUSH_F64 2\nNE\nASSERT\n'
        body+='ENUM_VAL 0 2\nPUSH_F64 3\nGT\nASSERT\n'
        self.paired(body)

    def test_mod_and_neg_retain_exact_type_refusal(self):
        for body in ('ENUM_VAL 0 2\nNEG\nPOP\n',
                     'ENUM_VAL 0 2\nPUSH_I64 1\nMOD\nPOP\n',
                     'PUSH_I64 2\nENUM_VAL 0 1\nMOD\nPOP\n'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix='nano-enum-refusal-') as tmp:
                work=Path(tmp);module=self.assemble(work,body)
                vm=self.run_command([ROOT/'bin/nano_vm',module])
                self.assertNotEqual(vm.returncode,0)
                self.assertIn('type error',vm.stderr)
                native=self.run_command([self.native(work,module,sanitize=True)])
                self.assertEqual(native.returncode,-signal.SIGABRT,native.stderr)
                self.assertIn('I stopped at a native invariant',native.stderr)
                for error in ('AddressSanitizer','LeakSanitizer','runtime error:'):
                    self.assertNotIn(error,native.stderr)

    def test_enum_return_checks_actual_tag(self):
        helpers=('.function identity 1 1 0 enum 1\n.parameters identity enum\n'
                 'LOAD_LOCAL 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-enum-return-') as tmp:
            work=Path(tmp);module=self.assemble(work,'PUSH_U8 2\nCALL identity\nPOP\n',helpers)
            native=self.run_command([self.native(work,module,sanitize=True)])
            self.assertEqual(native.returncode,-signal.SIGABRT,native.stderr)
            self.assertIn('I stopped at a native invariant',native.stderr)
            self.assertNotIn('Sanitizer',native.stderr)

    def test_invalid_definition_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-enum-definition-') as tmp:
            work=Path(tmp)
            source=work/'invalid.nasm'
            source.write_text('.types 0 1 0\n.entry main\n.function main 0 0 0 int 1\n'
                              'ENUM_VAL 1 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
            output=work/'previous.nvm';output.write_bytes(b'previous')
            result=self.run_command([ROOT/'bin/nanoisa','asm',source,'-o',output])
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_bytes(),b'previous')

    def test_enum_coercion_zero_and_wrapped_boundaries(self):
        body=('PUSH_I64 9223372036854775807\nENUM_VAL 0 1\nADD\n'
              'PUSH_I64 -9223372036854775808\nEQ\nASSERT\n'
              'PUSH_I64 -9223372036854775808\nENUM_VAL 0 1\nSUB\n'
              'PUSH_I64 9223372036854775807\nEQ\nASSERT\n'
              'PUSH_I64 -7\nENUM_VAL 0 2\nDIV\nPUSH_I64 -3\nEQ\nASSERT\n')
        for numerator in ('ENUM_VAL 0 65535','PUSH_I64 -7','PUSH_F64 nan','PUSH_F64 inf'):
            tag=3 if 'F64' in numerator else 1
            body+=numerator+f'\nENUM_VAL 0 0\nDIV\nDUP\nTYPE_CHECK {tag}\nASSERT\n'
            body+='CAST_STRING\nPUSH_STR zero\nEQ\nASSERT\n'
        self.paired(body)
