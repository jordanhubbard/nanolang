"""I retain the actual U8 result through ordinary native tail-call cleanup."""
from pathlib import Path
import signal
import tempfile
import unittest
from tests import test_native_tagged_arithmetic as tagged
ROOT=tagged.ROOT


class U8TailResults(unittest.TestCase):
    run_command=tagged.TaggedArithmetic.run_command
    checked=tagged.TaggedArithmetic.checked
    assemble=tagged.TaggedArithmetic.assemble
    native=tagged.TaggedArithmetic.native
    paired=tagged.TaggedArithmetic.paired

    def test_all_bytes_through_tail_chain_and_declaration_orders(self):
        helpers=[('.function identity 1 1 0 u8 1\n.parameters identity u8\n'
                  'LOAD_LOCAL 0\nRET\n.end\n'),
                 ('.function relay 1 1 0 u8 1\n.parameters relay u8\n'
                  'LOAD_LOCAL 0\nTAIL_CALL identity\n.end\n'),
                 ('.function outer 1 1 0 u8 1\n.parameters outer u8\n'
                  'LOAD_LOCAL 0\nTAIL_CALL relay\n.end\n')]
        body=''
        for value in range(256):
            body+=f'PUSH_U8 {value}\nCALL outer\nDUP\nTYPE_CHECK 2\nASSERT\n'
            body+=f'DUP\nCAST_INT\nPUSH_I64 {value}\nEQ\nASSERT\nPUSH_U8 {value}\nCALL identity\nEQ\nASSERT\n'
        for definitions in (helpers,list(reversed(helpers))):
            with self.subTest(reversed=definitions!=helpers):
                self.paired(body,''.join(definitions))

    def test_self_tail_staging_and_u8_result(self):
        helpers=('.function countdown 2 2 0 u8 1\n.parameters countdown u8 int\n'
                 'LOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_TRUE done\n'
                 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL countdown\n'
                 'done:\nLOAD_LOCAL 0\nRET\n.end\n')
        self.paired('PUSH_U8 255\nPUSH_I64 1000\nCALL countdown\nDUP\nTYPE_CHECK 2\nASSERT\n'
                    'CAST_INT\nPUSH_I64 255\nEQ\nASSERT\n',helpers)

    def test_tail_chain_preserves_exact_return_guard(self):
        helpers=('.function checked 1 1 0 u8 1\n.parameters checked u8\n'
                 'LOAD_LOCAL 0\nRET\n.end\n'
                 '.function relay 1 1 0 u8 1\n.parameters relay u8\n'
                 'LOAD_LOCAL 0\nTAIL_CALL checked\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-u8-tail-guard-') as tmp:
            work=Path(tmp)
            module=self.assemble(work,'PUSH_BOOL 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCALL relay\nPOP\n',helpers)
            vm=self.run_command([ROOT/'bin/nano_vm',module])
            self.assertNotEqual(vm.returncode,0)
            native=self.run_command([self.native(work,module,sanitize=True)])
            self.assertEqual(native.returncode,-signal.SIGABRT,native.stderr)
            self.assertIn('I stopped at a native invariant',native.stderr)
            for error in ('Sanitizer','runtime error:'):
                self.assertNotIn(error,native.stderr)
