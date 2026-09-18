"""I keep numeric-union storage explicit and exact scalar producers unchanged."""
from pathlib import Path
import tempfile
import unittest
from tests import test_native_tagged_arithmetic as tagged
ROOT = tagged.ROOT


class NumericUnion(unittest.TestCase):
    run_command = tagged.TaggedArithmetic.run_command
    checked = tagged.TaggedArithmetic.checked
    assemble = tagged.TaggedArithmetic.assemble
    native = tagged.TaggedArithmetic.native
    paired = tagged.TaggedArithmetic.paired
    boxed = tagged.TaggedArithmetic.boxed

    def test_concrete_and_boxed_join_orders(self):
        body=''
        for flag in (0,1):
            for reverse in (0,1):
                name=f'{flag}_{reverse}'
                integer='PUSH_I64 5\n'
                floating=self.boxed('PUSH_F64 3')+'PUSH_I64 2\nADD\n'
                first,second=(floating,integer) if reverse else (integer,floating)
                is_float=reverse if flag else not reverse
                body+=f'PUSH_BOOL {flag}\nJMP_FALSE other{name}\n'+first+f'JMP joined{name}\nother{name}:\n'+second+f'joined{name}:\n'
                body+=f'DUP\nTYPE_CHECK {3 if is_float else 1}\nASSERT\nPUSH_I64 5\nEQ\nASSERT\n'
        self.paired(body)

    def test_local_numeric_storage_and_loop_backedge(self):
        body=('PUSH_I64 1\nSTORE_LOCAL 0\nPUSH_I64 3\nSTORE_LOCAL 1\nloop:\n'
              'LOAD_LOCAL 0\nPUSH_F64 0.5\nADD\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nDUP\nSTORE_LOCAL 1\n'
              'JMP_TRUE loop\nLOAD_LOCAL 0\nDUP\nTYPE_CHECK 3\nASSERT\n'
              'PUSH_F64 2.5\nF64_EQ\nASSERT\n')
        self.paired(body)

    def test_loop_stack_numeric_storage(self):
        body=('PUSH_I64 3\nSTORE_LOCAL 1\nPUSH_I64 1\nloop:\n'
              'PUSH_F64 0.5\nADD\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\n'
              'DUP\nSTORE_LOCAL 1\nJMP_TRUE loop\nDUP\nTYPE_CHECK 3\nASSERT\n'
              'PUSH_F64 2.5\nF64_EQ\nASSERT\n')
        self.paired(body)

    def test_shared_calls_preserve_actual_parameter_tags(self):
        helper=('.function integer_tag 1 1 0 bool 1\n.parameters integer_tag int\n'
                'LOAD_LOCAL 0\nTYPE_CHECK 1\nRET\n.end\n')
        body=('PUSH_I64 4\nCALL integer_tag\nASSERT\n'
              'PUSH_F64 4\nCALL integer_tag\nBOOL_NOT\nASSERT\n')
        body+=self.boxed('PUSH_I64 3')+'PUSH_I64 1\nADD\nCALL integer_tag\nASSERT\n'
        body+=self.boxed('PUSH_F64 3')+'PUSH_I64 1\nADD\nCALL integer_tag\nBOOL_NOT\nASSERT\n'
        self.paired(body,helper)

    def test_optional_numeric_absence_keeps_void_tag(self):
        body=''
        for flag in (0,1):
            body+=f'PUSH_BOOL {flag}\nJMP_FALSE absent{flag}\n'+self.boxed('PUSH_F64 2')
            body+=f'PUSH_I64 1\nADD\nJMP joined{flag}\nabsent{flag}:\nPUSH_VOID\njoined{flag}:\n'
            body+=f'TYPE_CHECK {3 if flag else 0}\nASSERT\n'
        self.paired(body)

    def test_numeric_local_and_parameter_reject_unproved_payloads(self):
        numeric=self.boxed('PUSH_F64 2')+'PUSH_I64 1\nADD\n'
        helper=('.function inspect 1 1 0 bool 1\n.parameters inspect int\n'
                'LOAD_LOCAL 0\nTYPE_CHECK 1\nRET\n.end\n')
        for route in ('local','call'):
            with self.subTest(route=route), tempfile.TemporaryDirectory(prefix='nano-numeric-provenance-') as tmp:
                work=Path(tmp)
                body=(numeric+'STORE_LOCAL 0\nPUSH_U8 1\nSTORE_LOCAL 0\n' if route=='local' else
                      numeric+'CALL inspect\nPOP\nPUSH_U8 1\nCALL inspect\nPOP\n')
                module=self.assemble(work,body,helper if route=='call' else '')
                self.checked([ROOT/'bin/nano_vm',module])
                output=work/'previous.c';output.write_text('previous')
                result=self.run_command([ROOT/'bin/nvm2c',module,'-o',output])
                self.assertNotEqual(result.returncode,0)
                self.assertIn('provenance',result.stderr)
                self.assertEqual(output.read_text(),'previous')

    def test_non_numeric_join_provenance_is_not_widened(self):
        for wrong in ('PUSH_BOOL 1','PUSH_U8 1','PUSH_STR zero',self.boxed('PUSH_I64 1').strip()):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory(prefix='nano-union-refusal-') as tmp:
                work=Path(tmp)
                body='PUSH_BOOL 1\nJMP_FALSE other\n'+self.boxed('PUSH_F64 2')+'PUSH_I64 1\nADD\nJMP joined\nother:\n'+wrong+'\njoined:\nPOP\n'
                module=self.assemble(work,body)
                self.checked([ROOT/'bin/nano_vm',module])
                output=work/'previous.c';output.write_text('previous')
                result=self.run_command([ROOT/'bin/nvm2c',module,'-o',output])
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(output.read_text(),'previous')


if __name__ == '__main__':
    unittest.main()
