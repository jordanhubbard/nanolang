"""I preserve numeric tags through boxed arithmetic without widening heap shapes."""
from pathlib import Path
import signal
import tempfile
import unittest
from tests import test_native_floats as floats
ROOT = floats.ROOT


class TaggedArithmetic(unittest.TestCase):
    run_command = floats.NativeFloats.run_command
    checked = floats.NativeFloats.checked
    assemble = floats.NativeFloats.assemble
    native = floats.NativeFloats.native

    def paired(self, body, helpers=''):
        with tempfile.TemporaryDirectory(prefix='nano-boxed-numeric-') as tmp:
            work=Path(tmp); module=self.assemble(work,body,helpers)
            vm=self.checked([ROOT/'bin/nano_vm',module])
            native=self.checked([self.native(work,module,sanitize=True)])
            self.assertEqual(native.stdout,vm.stdout)

    def boxed(self, instruction, slot=0):
        return instruction+f'\nSTORE_GLOBAL {slot}\nLOAD_GLOBAL {slot}\n'

    def test_numeric_pairs_and_operand_order(self):
        body=''
        for left in ('I64','F64'):
            for right in ('I64','F64'):
                for boxed in ('left','right','both'):
                    floating='F64' in (left,right)
                    for op,result in [('ADD',9),('SUB',5),('MUL',14),('DIV',3.5 if floating else 3)]:
                        a=f'PUSH_{left} 7'; b=f'PUSH_{right} 2'
                        body+=(self.boxed(a) if boxed!='right' else a+'\n')
                        body+=(self.boxed(b,1) if boxed!='left' else b+'\n')
                        body+=f'{op}\nDUP\nTYPE_CHECK {3 if floating else 1}\nASSERT\n'
                        body+=f'PUSH_{"F64" if floating else "I64"} {result}\n{"F64_EQ" if floating else "I64_EQ"}\nASSERT\n'
        self.paired(body)

    def test_checked_locals_calls_and_tail_return(self):
        helpers=('.function compute 2 3 0 float 1\n.parameters compute float float\n'
                 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nADD\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nNEG\nRET\n.end\n'
                 '.function tail 2 2 0 float 1\n.parameters tail float float\n'
                 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nTAIL_CALL compute\n.end\n')
        body=''
        for a,b in [('PUSH_I64 7','PUSH_F64 2'),('PUSH_F64 7','PUSH_I64 2')]:
            body+=self.boxed(a)+self.boxed(b,1)+'CALL tail\nPUSH_F64 -9\nF64_EQ\nASSERT\n'
        self.paired(body,helpers)

    def test_boxed_branch_join_keeps_selected_numeric_tag(self):
        body=''
        for flag in (0,1):
            body+=f'PUSH_BOOL {flag}\nJMP_FALSE floating{flag}\n'
            body+=self.boxed('PUSH_I64 3')+'PUSH_I64 2\nADD\n'+f'JMP joined{flag}\nfloating{flag}:\n'
            body+=self.boxed('PUSH_F64 3')+'PUSH_I64 2\nADD\n'+f'joined{flag}:\n'
            body+='STORE_LOCAL 0\nLOAD_LOCAL 0\n'+f'TYPE_CHECK {1 if flag else 3}\nASSERT\n'
            body+='LOAD_LOCAL 0\nPUSH_I64 5\nEQ\nASSERT\n'
        self.paired(body)

    def test_numeric_boundaries_and_zero_totality(self):
        body=''
        for a,b,op,result in [(9223372036854775807,1,'ADD',-9223372036854775808),
                              (-9223372036854775808,1,'SUB',9223372036854775807),
                              (9223372036854775807,2,'MUL',-2),
                              (-9223372036854775808,-1,'DIV',-9223372036854775808),
                              (-9223372036854775808,-1,'MOD',0)]:
            body+=self.boxed(f'PUSH_I64 {a}')+self.boxed(f'PUSH_I64 {b}',1)+f'{op}\nPUSH_I64 {result}\nI64_EQ\nASSERT\n'
        body+=self.boxed('PUSH_I64 -9223372036854775808')+'NEG\nPUSH_I64 -9223372036854775808\nI64_EQ\nASSERT\n'
        body+=self.boxed('PUSH_I64 9007199254740993')+'PUSH_F64 0\nADD\nPUSH_F64 9007199254740992\nF64_EQ\nASSERT\n'
        body+=self.boxed('PUSH_F64 0')+'NEG\nCAST_STRING\nPUSH_STR negative\nEQ\nASSERT\n'
        for numerator in ('-7','nan','inf'):
            for zero in ('0','-0'):
                body+=self.boxed(f'PUSH_F64 {numerator}')+self.boxed(f'PUSH_F64 {zero}',1)+'DIV\nCAST_STRING\nPUSH_STR zero\nEQ\nASSERT\n'
        body+=self.boxed('PUSH_F64 nan')+'PUSH_I64 2\nADD\nDUP\nNE\nASSERT\n'
        self.paired(body)

    def test_invalid_tagged_operands_reach_exact_guard(self):
        for wrong in ('PUSH_BOOL 1','PUSH_U8 1','PUSH_VOID','PUSH_STR zero','PUSH_F64 1'):
            for op in ('ADD','SUB','MUL','DIV','MOD','NEG'):
                if wrong == 'PUSH_F64 1' and op != 'MOD':
                    continue
                with self.subTest(wrong=wrong,op=op), tempfile.TemporaryDirectory(prefix='nano-boxed-refusal-') as tmp:
                    work=Path(tmp)
                    body=self.boxed(wrong)+('' if op=='NEG' else self.boxed('PUSH_I64 2',1))+op+'\nPOP\n'
                    module=self.assemble(work,body)
                    self.assertNotEqual(self.run_command([ROOT/'bin/nano_vm',module]).returncode,0)
                    result=self.run_command([self.native(work,module,sanitize=True)])
                    self.assertEqual(result.returncode,-signal.SIGABRT,result.stdout+result.stderr)
                    for diagnostic in ('AddressSanitizer','LeakSanitizer','UndefinedBehaviorSanitizer','runtime error:'):
                        self.assertNotIn(diagnostic,result.stderr)

    def test_typed_consumers_keep_exact_result_tag_guards(self):
        for operand,consumer in [('PUSH_F64 2','I64_NEG'),('PUSH_I64 2','F64_NEG')]:
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory(prefix='nano-numeric-consumer-') as tmp:
                work=Path(tmp)
                module=self.assemble(work,self.boxed(operand)+'PUSH_I64 1\nADD\n'+consumer+'\nPOP\n')
                self.assertNotEqual(self.run_command([ROOT/'bin/nano_vm',module]).returncode,0)
                result=self.run_command([self.native(work,module,sanitize=True)])
                self.assertEqual(result.returncode,-signal.SIGABRT,result.stdout+result.stderr)
                for diagnostic in ('AddressSanitizer','LeakSanitizer','UndefinedBehaviorSanitizer','runtime error:'):
                    self.assertNotIn(diagnostic,result.stderr)

    def test_mixed_boxed_concrete_join_keeps_numeric_value(self):
        with tempfile.TemporaryDirectory(prefix='nano-numeric-shape-boundary-') as tmp:
            work=Path(tmp)
            body='PUSH_BOOL 1\nJMP_FALSE concrete\n'+self.boxed('PUSH_I64 2')
            body+='PUSH_F64 1.5\nADD\nJMP joined\nconcrete:\nPUSH_F64 3.5\njoined:\nDUP\nTYPE_CHECK 3\nASSERT\nPUSH_F64 3.5\nF64_EQ\nASSERT\n'
            module=self.assemble(work,body);self.checked([ROOT/'bin/nano_vm',module])
            self.checked([self.native(work,module,sanitize=True)])


if __name__ == '__main__':
    unittest.main()
