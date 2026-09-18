"""I match primitive exact-tag formatting and release original fallback owners."""
import unittest
from tests import test_llvm_managed_strings as managed
from tests import test_managed_binary64_format as binary_format
ROOT=managed.ROOT
class ManagedPrimitiveFormat(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node
    reference=binary_format.Binary64Format.reference

    def test_numeric_boundaries_reference_bytes_and_exact_tag_fallback(self):
        strings='.string zero "0"\n';body=''
        integers=[-(1<<63),(1<<63)-1,-42,-1,0,1,42]
        for i,value in enumerate(integers):
            strings+=f'.string int{i} "{value}"\n'
            body+=f'PUSH_I64 {value}\nCALL integer\nPUSH_STR int{i}\nEQ\nASSERT\n'
            body+=f'PUSH_I64 {value}\nSTR_FROM_FLOAT\nPUSH_STR zero\nEQ\nASSERT\n'
        for i,(_,expected,literal) in enumerate(self.reference()[:64]):
            strings+=f'.string float{i} "{expected}"\n'
            body+=f'PUSH_F64 {literal}\nCALL floating\nPUSH_STR float{i}\nEQ\nASSERT\n'
            body+=f'PUSH_F64 {literal}\nSTR_FROM_INT\nPUSH_STR zero\nEQ\nASSERT\n'
        for op in ('STR_FROM_INT','STR_FROM_FLOAT'):
            for value in ('PUSH_BOOL 1','PUSH_BOOL 0','PUSH_U8 255','PUSH_VOID','ENUM_VAL 0 42','PUSH_STR a'):
                body+=f'{value}\n{op}\nPUSH_STR zero\nEQ\nASSERT\n'
        body+='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\n'
        for name in ('fallback_integer','fallback_float'):
            body+=f'LOAD_GLOBAL 0\nCALL {name}\nPUSH_STR zero\nEQ\nASSERT\n'
            body+='LOAD_GLOBAL 0\nPUSH_STR a\nEQ\nASSERT\n'
        suffix=''
        for name,tag,op in [('integer','int','STR_FROM_INT'),('floating','float','STR_FROM_FLOAT'),
                            ('fallback_integer','string','STR_FROM_INT'),('fallback_float','string','STR_FROM_FLOAT')]:
            suffix+=f'.function {name} 1 1 0 string 1\n.parameters {name} {tag}\nLOAD_LOCAL 0\n{op}\nRET\n.end\n'
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);check(e.nms_module_live_bytes()===3n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_fallback_allocation_failure_releases_input_and_recovers(self):
        for op in ('STR_FROM_INT','STR_FROM_FLOAT'):
            body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL convert\nPOP\n'
            suffix=f'.function convert 1 1 0 string 1\n.parameters convert string\nLOAD_LOCAL 0\n{op}\nRET\n.end\n'
            _,ir,wasm=self.compile(self.program(body,suffix))
            extra='static long budget=-1;void *nano_test_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}'
            self.native_harness(ir,'budget=2;if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;budget=-1;for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,allocation_control=True)
            self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
if __name__=='__main__':unittest.main()
