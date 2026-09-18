"""I observe typed unary negation using exact integer representations."""
import unittest
from tests import test_reconstruction_f64_transport as transport
from tests import test_reconstruction_f64_comparisons as comparisons
from tests.test_canonical_f64_bits import PATTERNS
from scripts.nanoisa_reconstruction import Analyze, Expr, INT, FLOAT, BOOL, Refusal

class FloatNegation(unittest.TestCase):
    setUp=transport.FloatReconstruction.setUp
    command=transport.FloatReconstruction.command
    module=transport.FloatReconstruction.module
    native=transport.FloatReconstruction.native
    paired=transport.FloatReconstruction.paired
    observe_call_count=transport.FloatReconstruction.observe_call_count
    inspect=comparisons.FloatComparisons.inspect

    def test_sign_bit_original_and_double_negation(self):
        patterns=PATTERNS+(0x7fffffffffffffff,0xffffffffffffffff)
        for start in range(0,len(patterns),4):
            body='';ordinal=0
            for bits in patterns[start:start+4]:
                body+=f'PUSH_F64 bits:{bits:016x}\nSTORE_LOCAL 0\n'
                for operations,expected in (('F64_NEG\n',bits^(1<<63)),('',bits),('F64_NEG\nF64_NEG\n',bits)):
                    signed=expected if expected<1<<63 else expected-(1<<64)
                    body+=transport.FloatReconstruction.check(f'LOAD_LOCAL 0\n{operations}F64_TO_BITS\nPUSH_I64 {signed}\n',ordinal)
                    ordinal+=1
            self.paired('.entry main\n.function main 0 1 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
            self.inspect(['F64_NEG'])

    def test_calls_snapshots_branches_and_loop(self):
        text='''.entry main
.function main 0 2 0 int 1
PUSH_F64 bits:fff0000000000042
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
PUSH_F64 bits:0000000000000000
STORE_LOCAL 0
F64_NEG
F64_TO_BITS
PUSH_I64 9218868437227405378
I64_EQ
JMP_TRUE snapshot_ok
PUSH_I64 1
RET
snapshot_ok:
PUSH_F64 bits:7ff0000000000042
CALL relay
F64_NEG
POP
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 1
PUSH_I64 2
I64_LT_S
PUSH_F64 bits:8000000000000000
F64_NEG
F64_TO_BITS
PUSH_I64 0
I64_EQ
BOOL_AND
JMP_FALSE done
LOAD_LOCAL 1
PUSH_I64 0
I64_EQ
JMP_FALSE other
PUSH_F64 bits:0000000000000000
F64_NEG
STORE_LOCAL 0
JMP joined
other:
LOAD_LOCAL 0
F64_NEG
STORE_LOCAL 0
joined:
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
JMP_TRUE okay
PUSH_I64 2
RET
okay:
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
        self.inspect(['F64_NEG'])
        self.observe_call_count(c)

    def test_other_tags_refused(self):
        module={'functions':[{'params':[],'result':INT,'locals':0,'size':1,'code':[{'op':'F64_NEG','pc':0,'arg':0}]}]}
        for tag in (INT,BOOL,0,7):
            with self.assertRaisesRegex(Refusal,'exact scalar operand'):
                Analyze(module,0).simple(0,[Expr(tag,'constant',0)],set(),[])
        with self.assertRaisesRegex(Refusal,'populated scalar'):
            Analyze(module,0).simple(0,[],set(),[])
    test_remaining_float_operations_preserve_outputs=transport.FloatReconstruction.test_float_operations_remain_refused_without_publication

if __name__=='__main__':unittest.main()
