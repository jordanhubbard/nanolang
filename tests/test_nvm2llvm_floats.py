"""I preserve typed F64 and checked scalar casts in my shared LLVM lowering."""
import unittest
from tests import test_nvm2llvm as scalar


class LLVMFloats(unittest.TestCase):
    setUp = scalar.ScalarLLVM.setUp
    run_cmd = scalar.ScalarLLVM.run_cmd
    module = scalar.ScalarLLVM.module
    compare = scalar.ScalarLLVM.compare

    def program(self, body, suffix=''):
        return '.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'+suffix

    def test_arithmetic_comparisons_and_nan(self):
        body = ''
        for op, a, b, answer in (
            ('F64_ADD','1.25','2.5','3.75'), ('F64_SUB','5.5','2.25','3.25'),
            ('F64_MUL','1.5','-2.5','-3.75'), ('F64_DIV','7.5','2.5','3.0'),
            ('F64_DIV','2.5','0.0','0.0'), ('F64_DIV','-2.5','-0.0','0.0'),
        ):
            body += f'PUSH_F64 {a}\nPUSH_F64 {b}\n{op}\nPUSH_F64 {answer}\nF64_EQ\nASSERT\n'
        for op, a, b in (('F64_LT','1.0','2.0'),('F64_LE','2.0','2.0'),('F64_GT','2.0','1.0'),('F64_GE','2.0','2.0')):
            body += f'PUSH_F64 {a}\nPUSH_F64 {b}\n{op}\nASSERT\n'
        for op in ('F64_EQ','F64_LT','F64_LE','F64_GT','F64_GE'):
            body += f'PUSH_F64 nan\nPUSH_F64 1.0\n{op}\nBOOL_NOT\nASSERT\n'
        body += 'PUSH_F64 nan\nPUSH_F64 nan\nF64_NE\nASSERT\n'
        body += 'PUSH_F64 inf\nF64_NEG\nPUSH_F64 -inf\nF64_EQ\nASSERT\n'
        self.compare(self.program(body))

    def test_float_calls_locals_joins_truthiness_and_signed_zero(self):
        suffix = ('.function negate 1 1 0 float 1\n.parameters negate float\nLOAD_LOCAL 0\nF64_NEG\nRET\n.end\n'
                  '.function negative_zero 0 0 0 float 1\nPUSH_F64 -0.0\nRET\n.end\n'
                  '.function divide_zero 0 0 0 float 1\nPUSH_F64 -2.5\nPUSH_F64 -0.0\nF64_DIV\nRET\n.end\n')
        body = ('CALL divide_zero\nPUSH_F64 0.0\nF64_EQ\nASSERT\n'
                'PUSH_F64 3.5\nCALL negate\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_F64 -3.5\nF64_EQ\nASSERT\n'
                'CALL negative_zero\nDUP\nTYPE_CHECK 3\nASSERT\nJMP_FALSE zero\nPUSH_BOOL 0\nASSERT\nzero:\n'
                'PUSH_F64 nan\nJMP_TRUE truthy\nPUSH_BOOL 0\nASSERT\ntruthy:\n'
                'PUSH_BOOL 0\nJMP_TRUE other\nPUSH_F64 2.5\nJMP join\nother:\nPUSH_F64 -3.5\njoin:\n'
                'PUSH_F64 2.5\nF64_EQ\nASSERT\n')
        ir = self.compare(self.program(body, suffix))
        # The scalar ISA has no sign-bit opcode; inspect the translated helper's
        # actual returned bits through a normal LLVM harness, without fast math.
        probe = ir.read_text().replace('define i32 @main()', 'define i32 @original_main()')
        probe += ('define i32 @main() {\n %v = call %V @f2()\n %bits = extractvalue %V %v, 0\n'
                  ' %same = icmp eq i64 %bits, -9223372036854775808\n'
                  ' %z = call %V @f3()\n %zbits = extractvalue %V %z, 0\n'
                  ' %positive_zero = icmp eq i64 %zbits, 0\n %both = and i1 %same, %positive_zero\n'
                  ' %status = select i1 %both, i32 0, i32 1\n ret i32 %status\n}\n')
        ir.write_text(probe)
        self.run_cmd(['opt','-passes=default<O2>',ir,'-o',self.work/'zero.bc'])
        self.run_cmd(['lli',self.work/'zero.bc'])

    def test_checked_integer_and_float_conversions(self):
        body = ''
        for value, answer in (('3.75',3),('-3.75',-3),('-0.0',0),('-9223372036854775808.0',-9223372036854775808),('9223372036854774784.0',9223372036854774784)):
            body += f'PUSH_F64 {value}\nCAST_INT\nPUSH_I64 {answer}\nI64_EQ\nASSERT\n'
        for instruction, answer in (('PUSH_I64 -9','-9.0'),('PUSH_BOOL 1','1.0'),('PUSH_F64 -3.5','-3.5')):
            body += f'{instruction}\nCAST_FLOAT\nPUSH_F64 {answer}\nF64_EQ\nASSERT\n'
        self.compare(self.program(body))

    def test_invalid_float_to_int_traps_before_conversion(self):
        for value in ('nan','inf','-inf','9223372036854775808.0','-9223372036854777856.0'):
            with self.subTest(value=value):
                self.compare(self.program(f'PUSH_F64 {value}\nCAST_INT\nPOP\n'), trap=True)

    def test_float_entry_is_outside_executable_contract(self):
        module = self.module('.entry main\n.function main 0 0 0 float 1\nPUSH_F64 1.0\nRET\n.end\n')
        self.run_cmd([scalar.VM,'--verify-only',module])
        result = self.run_cmd([scalar.LLVM,module], success=False)
        self.assertIn('integer/bool executable', result.stderr)


if __name__ == '__main__':
    unittest.main()
