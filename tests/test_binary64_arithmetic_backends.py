"""I compare exact scalar arithmetic result bits across the same-module backends."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_nvm2wasm as wasm
from tests import test_llvm_managed_strings as managed

Q=0x7ff8000000000000
ONE=0x3ff0000000000000
INF=0x7ff0000000000000
SIGN=1<<63
NANS=(0x7ff0000000000001,0xfff0000000000042,0x7ff8123456789abc,0xfff8000000001234)
def observe(expected):
    signed=expected if expected<SIGN else expected-(1<<64)
    return f'F64_TO_BITS\nPUSH_I64 {signed}\nI64_EQ\nASSERT\n'
def operands(a,b,boxed=False):
    if boxed:
        return f'PUSH_F64 bits:{a:016x}\nSTORE_GLOBAL 0\nPUSH_F64 bits:{b:016x}\nSTORE_GLOBAL 1\nLOAD_GLOBAL 0\nLOAD_GLOBAL 1\n'
    return f'PUSH_F64 bits:{a:016x}\nPUSH_F64 bits:{b:016x}\n'

class ArithmeticBackends(unittest.TestCase):
    module=wasm.ScalarWasm.module
    compare=wasm.ScalarWasm.compare
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-arithmetic-backends-'))
        self.sequence=0
    def run_cmd(self,args,success=True):
        args=list(map(str,args))
        if args[0]=='cc':
            args=shlex.split(os.environ.get('CC','cc'))+args[1:]
            if any(a.endswith('.c') for a in args):
                args+=['-ffp-contract=off','-fno-fast-math','-fsanitize=address,undefined','-fno-sanitize-recover=all']
        self.sequence+=1
        p=subprocess.run(args,capture_output=True,text=True,timeout=90)
        (self.work/f'{self.sequence}.log').write_text(shlex.join(args)+'\n'+p.stdout+p.stderr)
        self.assertEqual(p.returncode==0,success,p.stdout+p.stderr+str(self.work))
        return p
    def program(self,body,locals=0):
        return f'.entry main\n.function main 0 {locals} 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'
    def test_nan_typed_known_and_boxed(self):
        for typed in ('F64_ADD','F64_SUB','F64_MUL','F64_DIV'):
            for route in ('typed','known','boxed'):
                op=typed if route=='typed' else typed[4:]
                body=''
                for nan in NANS:
                    for a,b in ((nan,ONE),(ONE,nan),(nan,NANS[-1])):
                        body+=operands(a,b,route=='boxed')+op+'\nDUP\nTYPE_CHECK 3\nASSERT\n'
                        if route=='boxed':body+='CAST_FLOAT\n'
                        body+=observe(Q)
                ir=self.compare(self.program(body))
                self.assertIn('@float_result',ir.read_text())
                self.assertNotIn(' fast ',ir.read_text())
    def test_zero_precedence_invalid_operations_and_rounding(self):
        cases=[('F64_ADD',INF,INF|SIGN,Q),('F64_SUB',INF,INF,Q),('F64_MUL',0,INF,Q),('F64_DIV',INF,INF,Q),
               ('F64_ADD',SIGN,SIGN,SIGN),('F64_ADD',SIGN,0,0),('F64_MUL',SIGN,ONE,SIGN),
               ('F64_ADD',ONE,0x3ca0000000000000,ONE),('F64_ADD',ONE+1,0x3ca0000000000000,ONE+2),
               ('F64_MUL',1,0x3fe0000000000000,0),('F64_MUL',3,0x3fe0000000000000,2),
               ('F64_MUL',SIGN|1,0x3fe0000000000000,SIGN),('F64_ADD',1,1,2),
               ('F64_ADD',0x7fefffffffffffff,0x7fefffffffffffff,INF),('F64_DIV',ONE,0x4008000000000000,0x3fd5555555555555)]
        cases += [('F64_DIV',a,b,0) for a in (ONE,INF,*NANS) for b in (0,SIGN)]
        for start in range(0,len(cases),8):
            body=''.join(operands(a,b)+op+'\n'+observe(want) for op,a,b,want in cases[start:start+8])
            self.compare(self.program(body))
        self.compare(self.program(operands(ONE+1,0x3feffffffffffffe)+'F64_MUL\n'+f'PUSH_F64 bits:{ONE:016x}\nF64_SUB\n'+observe(0)))
    def test_mixed_promotion_and_input_transport_unchanged(self):
        for op in ('ADD','SUB','MUL','DIV'):
            body=''
            for nan in NANS:
                body+=f'PUSH_F64 bits:{nan:016x}\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 1\n{op}\n'+observe(Q)
                body+=f'PUSH_I64 1\nLOAD_LOCAL 0\n{op}\n'+observe(Q)
                body+='LOAD_LOCAL 0\n'+observe(nan)
                body+='LOAD_LOCAL 0\nF64_NEG\n'+observe(nan^SIGN)
            self.compare(self.program(body,1))

class ManagedArithmetic(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node
    def test_managed_scalar_cleanup(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 0\n'
        for op in ('F64_ADD','F64_SUB','F64_MUL','F64_DIV'):
            body+=operands(NANS[0],ONE)+op+'\n'+observe(Q)
        _,ir,module=self.compile(self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(module,'check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',module]).stdout,'0\n')

if __name__=='__main__':unittest.main()
