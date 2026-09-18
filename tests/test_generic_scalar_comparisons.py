"""I preserve generic tagged comparison compatibility, including unordered NaN."""
import unittest
from tests import test_nvm2llvm as llvm
from tests import test_nvm2wasm as wasm


class GenericScalarComparisons(unittest.TestCase):
    setUp = wasm.ScalarWasm.setUp
    run_cmd = wasm.ScalarWasm.run_cmd
    module = wasm.ScalarWasm.module
    compare = wasm.ScalarWasm.compare

    def program(self, body, suffix=''):
        return '.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+suffix

    def check(self, left, right, equal, order):
        body = ''
        for op, expected in [('EQ', equal), ('NE', not equal), ('LT', order < 0),
                             ('LE', order <= 0), ('GT', order > 0), ('GE', order >= 0)]:
            body += f'{left}\n{right}\n{op}\nDUP\nTYPE_CHECK 4\nASSERT\n'
            body += ('BOOL_NOT\n' if not expected else '')+'ASSERT\n'
        return body

    def test_all_scalar_tag_pairs(self):
        values = ['PUSH_VOID', 'PUSH_I64 0', 'PUSH_U8 0', 'PUSH_F64 0.0', 'PUSH_BOOL 0']
        equality = [[1,0,0,0,0], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [0,0,0,0,1]]
        order = [[0,-1,-1,-1,-1], [1,0,-1,0,-1], [1,1,0,-1,-1],
                 [1,0,1,0,-1], [1,1,1,1,0]]
        self.compare(self.program(''.join(self.check(a,b,bool(equality[i][j]),order[i][j])
                     for i,a in enumerate(values) for j,b in enumerate(values))))

    def test_numeric_boundaries_and_cross_tag_order(self):
        cases = [('PUSH_I64 -9223372036854775808', 'PUSH_I64 9223372036854775807', False, -1),
                 ('PUSH_U8 127', 'PUSH_U8 128', False, -1),
                 ('PUSH_BOOL 0', 'PUSH_BOOL 1', False, -1),
                 ('PUSH_U8 0', 'PUSH_I64 100', False, 1),
                 ('PUSH_BOOL 0', 'PUSH_F64 nan', False, 1),
                 ('PUSH_F64 -0.0', 'PUSH_F64 0.0', True, 0),
                 ('PUSH_F64 -inf', 'PUSH_F64 inf', False, -1),
                 ('PUSH_F64 inf', 'PUSH_F64 inf', True, 0),
                 ('PUSH_F64 -3.5', 'PUSH_I64 -3', False, -1)]
        body = ''
        for a,b,equal,order in cases:
            body += self.check(a,b,equal,order)+self.check(b,a,equal,-order)
        self.compare(self.program(body))

    def test_mixed_integer_float_retains_binary64_rounding(self):
        cases = [(9007199254740993, '9007199254740992.0', True, 0),
                 (9007199254740991, '9007199254740992.0', False, -1),
                 (-9007199254740993, '-9007199254740992.0', True, 0),
                 (9223372036854775807, '9223372036854775808.0', True, 0),
                 (-9223372036854775808, '-9223372036854775808.0', True, 0)]
        body = ''
        for integer,floating,equal,order in cases:
            a,b=f'PUSH_I64 {integer}',f'PUSH_F64 {floating}'
            body += self.check(a,b,equal,order)+self.check(b,a,equal,-order)
        self.compare(self.program(body))

    def test_generic_nan_order_is_distinct_from_equality_and_typed_f64(self):
        body = ''
        for other in ('PUSH_F64 nan', 'PUSH_F64 inf', 'PUSH_F64 -0.0', 'PUSH_I64 17'):
            body += self.check('PUSH_F64 nan',other,False,0)
            body += self.check(other,'PUSH_F64 nan',False,0)
        for op in ('F64_EQ','F64_LT','F64_LE','F64_GT','F64_GE','F64_NE'):
            body += f'PUSH_F64 nan\nPUSH_F64 0.0\n{op}\n'
            body += ('BOOL_NOT\n' if op != 'F64_NE' else '')+'ASSERT\n'
        self.compare(self.program(body))

    def test_calls_locals_and_boolean_result_join(self):
        suffix = ('.function order 2 2 0 bool 1\n.parameters order u8 float\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 1\nLT\nRET\n.end\n')
        body = ('PUSH_U8 255\nPUSH_F64 -100.0\nCALL order\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nJMP_FALSE bad\nPUSH_I64 1\nPUSH_F64 1.0\nEQ\nJMP joined\n'
                'bad:\nPUSH_VOID\nPUSH_BOOL 0\nEQ\njoined:\nSTORE_LOCAL 1\n'
                'LOAD_LOCAL 1\nTYPE_CHECK 4\nASSERT\nLOAD_LOCAL 1\nASSERT\n')
        self.compare(self.program(body,suffix))

    def test_comparison_operands_are_eager(self):
        suffix = ('.function right 0 0 0 float 1\nPUSH_BOOL 0\nASSERT\nPUSH_F64 0.0\nRET\n.end\n')
        self.compare(self.program('PUSH_BOOL 0\nCALL right\nEQ\nPOP\n',suffix),trap=True)

    def test_heap_refusal_preserves_output(self):
        for prefix,body in [('.string text "text"\n','PUSH_STR text\nPUSH_I64 0\nPUSH_I64 1\nSTR_SUBSTR\nPOP\n')]:
            module = self.module(prefix+self.program(body))
            self.run_cmd([llvm.VM,module])
            for translator in (llvm.LLVM,wasm.WASM):
                out=self.work/'previous'
                out.write_text('previous')
                self.run_cmd([translator,module,'-o',out],success=False)
                self.assertEqual(out.read_text(),'previous')


if __name__ == '__main__':
    unittest.main()
