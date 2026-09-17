"""I preserve eager scalar truthiness on one module across all four backends."""
import unittest
from tests import test_nvm2llvm as llvm
from tests import test_nvm2wasm as wasm


class ScalarTruthiness(unittest.TestCase):
    setUp = wasm.ScalarWasm.setUp
    run_cmd = wasm.ScalarWasm.run_cmd
    module = wasm.ScalarWasm.module
    compare = wasm.ScalarWasm.compare

    def program(self, body, suffix=''):
        return '.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+suffix

    def test_scalar_truth_tables_and_result_tags(self):
        values = [('PUSH_VOID',False),('PUSH_I64 0',False),('PUSH_I64 -7',True),
                  ('PUSH_BOOL 0',False),('PUSH_BOOL 1',True),('PUSH_F64 0.0',False),
                  ('PUSH_F64 -0.0',False),('PUSH_F64 -1.25',True),
                  ('PUSH_F64 nan',True),('PUSH_F64 inf',True)]
        body = ''
        for instruction, truth in values:
            for op, expected in [('CAST_BOOL',truth),('NOT',not truth)]:
                body += f'{instruction}\n{op}\nDUP\nTYPE_CHECK 4\nASSERT\n'
                body += ('BOOL_NOT\n' if not expected else '')+'ASSERT\n'
        for a, ta in values:
            for b, tb in values:
                for op, expected in [('AND',ta and tb),('OR',ta or tb)]:
                    body += f'{a}\n{b}\n{op}\nDUP\nTYPE_CHECK 4\nASSERT\n'
                    body += ('BOOL_NOT\n' if not expected else '')+'ASSERT\n'
        self.compare(self.program(body))

    def test_calls_locals_and_join_keep_bool_tag(self):
        suffix = ('.function truth 1 1 0 bool 1\n.parameters truth float\n'
                  'LOAD_LOCAL 0\nCAST_BOOL\nRET\n.end\n')
        body = ('PUSH_F64 nan\nCALL truth\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nTYPE_CHECK 4\nASSERT\n'
                'LOAD_LOCAL 0\nJMP_TRUE yes\nPUSH_F64 0.0\nCAST_BOOL\nJMP join\nyes:\n'
                'PUSH_F64 -0.0\nNOT\njoin:\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nTYPE_CHECK 4\nASSERT\n'
                'LOAD_LOCAL 1\nPUSH_I64 5\nAND\nASSERT\n')
        self.compare(self.program(body,suffix))

    def test_void_tag_survives_local_storage(self):
        self.compare(self.program('PUSH_VOID\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nDUP\nTYPE_CHECK 0\nASSERT\nCAST_BOOL\nBOOL_NOT\nASSERT\n'))

    def test_tagged_void_integer_join_retains_truthiness(self):
        suffix = ('.function choose 1 1 0 bool 1\n.parameters choose bool\n'
                  'LOAD_LOCAL 0\nJMP_TRUE present\nPUSH_VOID\nJMP join\npresent:\n'
                  'PUSH_I64 7\njoin:\nCAST_BOOL\nRET\n.end\n')
        self.compare(self.program('PUSH_BOOL 0\nCALL choose\nBOOL_NOT\nASSERT\n'
                                  'PUSH_BOOL 1\nCALL choose\nASSERT\n',suffix))

    def test_logical_ops_consume_eager_call_results(self):
        # Earlier CALL instructions must run even when a logical result could
        # otherwise be determined from the first value alone.
        suffix = ('.function checked_right 0 0 0 bool 1\n'
                  'PUSH_BOOL 0\nASSERT\nPUSH_BOOL 1\nRET\n.end\n')
        for op, left in [('AND',0),('OR',1)]:
            with self.subTest(op=op):
                self.compare(self.program(f'PUSH_BOOL {left}\nCALL checked_right\n{op}\nPOP\n',suffix),trap=True)

    def test_heap_stays_outside_new_profile(self):
        for prefix, value in [('.string text "text"\n','PUSH_STR text')]:
            for op in ('CAST_BOOL','NOT','AND','OR'):
                with self.subTest(value=value,op=op):
                    body=value+'\n'+('PUSH_BOOL 1\n' if op in ('AND','OR') else '')+op+'\nPOP\n'
                    module=self.module(prefix+self.program(body))
                    self.run_cmd([llvm.VM,'--verify-only',module])
                    self.run_cmd([llvm.VM,module])
                    for translator in (llvm.C,llvm.LLVM,wasm.WASM):
                        target=self.work/'retained-output'
                        target.write_bytes(b'previous')
                        self.run_cmd([translator,module,'-o',target],success=False)
                        self.assertEqual(target.read_bytes(),b'previous')


if __name__ == '__main__':
    unittest.main()
