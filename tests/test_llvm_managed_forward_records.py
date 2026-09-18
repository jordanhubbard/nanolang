"""I execute exact forward nominal DAGs without widening unknown/mixed shapes."""
import unittest
from tests import test_llvm_managed_records as records
from tests.test_managed_record_shapes import program, function, NO
ROOT=records.ROOT

class ForwardRecords(unittest.TestCase):
    setUp=records.ManagedRecords.setUp
    run_cmd=records.ManagedRecords.run_cmd
    compile=records.ManagedRecords.compile
    native_harness=records.ManagedRecords.native_harness
    node=records.ManagedRecords.node
    paired=records.ManagedRecords.paired

    def test_original_forward_and_diamond_aliases_exact_bits(self):
        self.paired(program('PUSH_I64 1\nSTRUCT_LITERAL 1 1\nSTRUCT_LITERAL 0 1\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 1\nEQ\nASSERT',[(0,[(8,1)]),(0,[(1,NO)])]))
        layouts=[(0,[(8,1),(8,2)]),(0,[(8,4)]),(0,[(8,4)]),(0,[]),
                 (0,[(1,NO),(3,NO),(4,NO),(5,NO),(2,NO)])]
        body='STRUCT_NEW 3\nPOP\n'
        for bits in (-9223372036854775808,1,9221120237041090611):
            body+=(f'PUSH_I64 42\nPUSH_I64 {bits}\nF64_FROM_BITS\nPUSH_BOOL 1\nPUSH_STR text\nPUSH_U8 255\nSTRUCT_LITERAL 4 5\nSTORE_LOCAL 0\n'
                   'LOAD_LOCAL 0\nSTRUCT_LITERAL 1 1\nLOAD_LOCAL 0\nSTRUCT_LITERAL 2 1\nAGG_PACK 0 0 0 2\nSTORE_LOCAL 1\n'
                   'LOAD_LOCAL 1\nAGG_GET 0\nAGG_GET 0\nLOAD_LOCAL 1\nAGG_GET 1\nAGG_GET 0\nEQ\nASSERT\n'
                   f'LOAD_LOCAL 1\nAGG_GET 1\nAGG_GET 0\nAGG_GET 1\nF64_TO_BITS\nPUSH_I64 {bits}\nEQ\nASSERT\n'
                   'LOAD_LOCAL 0\nPUSH_STR empty\nSTRUCT_SET 3\nPOP\n'
                   'LOAD_LOCAL 1\nAGG_GET 0\nAGG_GET 0\nAGG_GET 3\nPUSH_STR empty\nEQ\nASSERT\n'
                   'LOAD_LOCAL 0\nSTRUCT_LITERAL 1 1\nLOAD_LOCAL 1\nCALL replace\nLOAD_LOCAL 1\nEQ\nASSERT\n')
        helper=function('replace','LOAD_LOCAL 1\nLOAD_LOCAL 0\nAGG_SET 0\nRET',[8,8],2,8)
        self.paired(program(body,layouts,[8,8],[helper]))

    def test_permuted_chain_get_temporary_and_bounded_live_churn(self):
        layouts=[(0,[(8,3)]),(0,[(5,NO)]),(0,[(8,1)]),(0,[(8,2)])]
        body=('PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nPUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\n'
              'STRUCT_LITERAL 1 1\nSTRUCT_LITERAL 2 1\nSTRUCT_LITERAL 3 1\nSTRUCT_LITERAL 0 1\n'
              'AGG_GET 0\nAGG_GET 0\nAGG_GET 0\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 0\nPUSH_I64 12000\nLT\nJMP_TRUE loop\n')
        self.paired(program(body,layouts,[1]))

    def test_forward_global_vm_reentry_and_allocation_cleanup(self):
        layouts=[(0,[(8,1)]),(0,[(1,NO)])]
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE existing\n'
              'PUSH_I64 0\nSTRUCT_LITERAL 1 1\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\n'
              'existing:\nLOAD_GLOBAL 0\nAGG_GET 0\nDUP\nAGG_GET 0\nPUSH_I64 1\nI64_ADD\nSTRUCT_SET 0\nAGG_GET 0\nRET\n')
        module,ir,wasm=self.compile(program(body,layouts),vm_ok=False)
        self.run_cmd([ROOT/'obj/managed_record_reentry',module])
        self.native_harness(ir,'for(int i=1;i<4;i++)if(nano_try_entry()!=(uint64_t)i||nms_module_live_objects()!=2)return 1;return nano_dispose();')
        self.node(wasm,'for(let j=0;j<2;j++){e=new WebAssembly.Instance(m).exports;for(let i=1;i<4;i++)check(e.nano_try_entry()===BigInt(i));check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);}')
        body=('PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 1 1\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\n'
              'LOAD_GLOBAL 0\nAGG_GET 0\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\nPUSH_VOID\nSTORE_GLOBAL 0\n')
        _,ir,wasm=self.compile(program(body,[(0,[(8,1)]),(0,[(5,NO)])]))
        extra='static long budget=-1;void *nano_test_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}'
        for budget in (0,1,2,3,4,5,6,7,8,12):
            self.native_harness(ir,f'budget={budget};uint64_t s=nano_try_entry();if(s && s!=((uint64_t)3<<32))return 1;if(nms_module_live_objects())return 2;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 3;return nano_dispose();',extra,allocation_control=True)
        self.node(wasm,'for(let i=0;i<5;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')

    def test_unknown_and_wrong_forward_nominal_preserve_output(self):
        layouts=[(0,[(8,2)]),(0,[(1,NO)]),(0,[(1,NO)])]
        bads=[program('PUSH_I64 9\nSTRUCT_LITERAL 1 1\nSTRUCT_LITERAL 0 1\nPOP',layouts),
              program('PUSH_I64 9\nSTRUCT_LITERAL 2 1\nSTRUCT_LITERAL 0 1\nPOP',layouts,authority=False),
              program('PUSH_I64 9\nSTRUCT_LITERAL 2 1\nSTRUCT_LITERAL 0 1\nPOP',layouts,flags=[0,0,0])]
        for text in bads:
            source,module=self.work/'refuse.nasm',self.work/'refuse.nvm'
            source.write_text(text);self.run_cmd([ROOT/'bin/nanoisa','asm',source,'-o',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                out=self.work/'prior';out.write_bytes(b'prior output')
                self.run_cmd([ROOT/'bin'/tool,module,'-o',out],success=False)
                self.assertEqual(out.read_bytes(),b'prior output')
if __name__=='__main__':unittest.main()
