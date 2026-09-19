"""I retain scalar bit payloads beside managed allocations and cleanup."""
import unittest
from tests import test_llvm_managed_strings as managed
from tests.test_canonical_f64_bits import PATTERNS
class ManagedBinary64Bits(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node
    def test_bit_patterns_survive_managed_frames(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 0\n'
        for bits in PATTERNS:
            signed=bits if bits<(1<<63) else bits-(1<<64)
            body+=f'PUSH_I64 {signed}\nF64_FROM_BITS\nCALL relay\nF64_TO_BITS\nPUSH_I64 {signed}\nI64_EQ\nASSERT\n'
        suffix='.function relay 1 1 0 float 1\n.parameters relay float\nLOAD_LOCAL 0\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<3;i++){if(nano_try_entry()||nms_module_live_objects())return 1;}return nano_dispose();')
        self.node(wasm,'for(let i=0;i<3;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')
if __name__=='__main__':unittest.main()
