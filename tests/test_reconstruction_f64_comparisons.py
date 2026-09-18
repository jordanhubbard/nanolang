"""I compare typed binary64 values and observe retained bits as integers."""
import unittest
from tests import test_reconstruction_f64_transport as transport
ROOT=transport.ROOT
from scripts.nanoisa_reconstruction import Analyze, Expr, FLOAT, BOOL, INT, Refusal, FLOAT_COMPARE

class FloatComparisons(unittest.TestCase):
    setUp=transport.FloatReconstruction.setUp
    command=transport.FloatReconstruction.command
    module=transport.FloatReconstruction.module
    native=transport.FloatReconstruction.native
    paired=transport.FloatReconstruction.paired
    observe_call_count=transport.FloatReconstruction.observe_call_count

    def inspect(self, ops):
        for producer in ('nano_virt','nanoc_stage1','nanoc_stage2'):
            text=self.command([ROOT/'bin/nanoisa','dump',self.work/(producer+'.nvm')]).stdout
            for op in ops:
                self.assertIn(op,text)

    def test_ieee_results_and_unchanged_operand_bits(self):
        # Each row states ordering explicitly; None denotes unordered.
        pairs=[(0,0x8000000000000000,0),(1,0,1),
               (0x8000000000000001,0,-1),
               (0x000fffffffffffff,0x0010000000000000,-1),
               (0x7fefffffffffffff,0x7ff0000000000000,-1),
               (0xfff0000000000000,0xffefffffffffffff,-1),
               (0x7ff8000000000042,0x3ff0000000000000,None),
               (0xfff8000000000042,0xfff8000000000042,None),
               (0x7ff0000000000042,0x7ff0000000000000,None),
               (0xfff0000000000042,0x7ff8000000000043,None)]
        for op in FLOAT_COMPARE:
            body=''; ordinal=0
            for a,b,order in pairs:
                for left,right,relation in ((a,b,order),(b,a,None if order is None else -order)):
                    result={'F64_EQ':relation==0,'F64_NE':relation!=0,
                            'F64_LT':relation is not None and relation<0,
                            'F64_LE':relation is not None and relation<=0,
                            'F64_GT':relation is not None and relation>0,
                            'F64_GE':relation is not None and relation>=0}[op]
                    body+=f'PUSH_F64 bits:{left:016x}\nSTORE_LOCAL 0\nPUSH_F64 bits:{right:016x}\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n{op}\n'
                    body+=f'JMP_{"TRUE" if result else "FALSE"} okay{ordinal}\nPUSH_I64 1\nRET\nokay{ordinal}:\n';ordinal+=1
                    for slot,bits in enumerate((left,right)):
                        signed=bits if bits<1<<63 else bits-(1<<64)
                        body+=transport.FloatReconstruction.check(f'LOAD_LOCAL {slot}\nF64_TO_BITS\nPUSH_I64 {signed}\n',ordinal);ordinal+=1
            self.paired('.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
            self.inspect([op])

    def test_snapshots_calls_branches_and_pure_loop(self):
        text='''.entry main
.function main 0 2 0 int 1
PUSH_F64 bits:7ff0000000000042
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
PUSH_F64 bits:0000000000000000
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
F64_NE
JMP_TRUE good
PUSH_I64 1
RET
good:
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 1
PUSH_I64 2
I64_LT_S
PUSH_F64 bits:8000000000000000
PUSH_F64 bits:0000000000000000
F64_EQ
BOOL_AND
JMP_FALSE done
LOAD_LOCAL 0
PUSH_F64 bits:0000000000000000
F64_GT
JMP_FALSE other
PUSH_I64 9
RET
other:
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 0
F64_TO_BITS
PUSH_I64 0
I64_EQ
JMP_TRUE bits_ok
PUSH_I64 2
RET
bits_ok:
PUSH_I64 0
RET
.end
.function relay 1 1 0 float 1
.parameters relay float
LOAD_LOCAL 0
RET
.end
'''
        _,c=self.paired(text,'shadow nlr_f1_relay { assert (== (float_to_bits (nlr_f1_relay (float_from_bits 1))) 1) }')
        self.inspect(['F64_EQ','F64_NE','F64_GT'])
        self.observe_call_count(c)

    def test_exact_operand_boundaries(self):
        for op in FLOAT_COMPARE:
            module={'functions':[{'params':[],'result':INT,'locals':0,'size':1,'code':[{'op':op,'pc':0,'arg':0}]}]}
            for tag in (INT,BOOL,0,7):
                for index in (0,1):
                    stack=[Expr(FLOAT,'float_bits',0),Expr(FLOAT,'float_bits',0)]
                    stack[index]=Expr(tag,'constant',0)
                    with self.assertRaises(Refusal):
                        Analyze(module,0).simple(0,stack,set(),[])
    test_other_float_operations_preserve_outputs=transport.FloatReconstruction.test_float_operations_remain_refused_without_publication

if __name__=='__main__':unittest.main()
