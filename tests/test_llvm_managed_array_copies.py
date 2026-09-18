"""I qualify counted literal owners and fresh shallow copies on actual targets."""
import unittest
from tests import test_llvm_managed_mutable_arrays as arrays
ROOT=arrays.ROOT

class ArrayCopies(unittest.TestCase):
    setUp=arrays.MutableArrays.setUp
    run_cmd=arrays.MutableArrays.run_cmd
    program=arrays.MutableArrays.program
    compile=arrays.MutableArrays.compile
    native_harness=arrays.MutableArrays.native_harness
    node=arrays.MutableArrays.node
    paired=arrays.MutableArrays.paired

    def test_literal_matrix_counts_and_slice_order(self):
        body=''
        pairs=[(1,'PUSH_I64 -9','PUSH_I64 -9'),(1,'PUSH_U8 255','PUSH_I64 255'),
               (2,'PUSH_I64 -1','PUSH_U8 255'),(2,'PUSH_U8 254','PUSH_U8 254'),
               (3,'PUSH_F64 -0.0','PUSH_F64 -0.0'),(3,'PUSH_I64 9007199254740993','PUSH_F64 9007199254740992'),
               (4,'PUSH_BOOL 1','PUSH_BOOL 1')]
        for kind,value,expected in pairs:
            for count in (0,1,9,17):
                body+=(value+'\n')*count+f'ARR_LITERAL {kind} {count}\nSTORE_LOCAL 0\n'
                body+=f'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 {count}\nEQ\nASSERT\n'
                body+='LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 4294967295\nARR_SLICE\nSTORE_LOCAL 1\n'
                body+=f'LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 {count}\nEQ\nASSERT\n'
                if count:body+=f'LOAD_LOCAL 1\nARR_POP\nDUP\nTYPE_CHECK {kind}\nASSERT\n{expected}\nEQ\nASSERT\n'
                body+='PUSH_VOID\nSTORE_LOCAL 0\nPUSH_VOID\nSTORE_LOCAL 1\n'
        self.paired(body)

    def test_boxed_leaf_order_duplicates_and_copy_independence(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 2\n'
        values=[('PUSH_VOID',0),('PUSH_I64 71',1),('PUSH_U8 255',2),('PUSH_F64 0.5',3),('PUSH_BOOL 1',4),('LOAD_LOCAL 2',5),('ENUM_VAL 0 17',9),('LOAD_LOCAL 2',5)]
        for kind in (0,5,9):
            body+='\n'.join(v for v,_ in values)+f'\nARR_LITERAL {kind} 8\nSTORE_LOCAL 0\n'
            body+='LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 8\nARR_SLICE\nSTORE_LOCAL 1\n'
            body+='LOAD_LOCAL 0\nLOAD_LOCAL 1\nEQ\nNOT\nASSERT\n'
            for i,(value,tag) in enumerate(values):
                body+=f'LOAD_LOCAL 1\nPUSH_I64 {i}\nARR_GET\nDUP\nTYPE_CHECK {tag}\nASSERT\n{value}\nEQ\nASSERT\n'
            body+='LOAD_LOCAL 1\nPUSH_I64 1\nPUSH_BOOL 0\nARR_SET\nPOP\nLOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nPUSH_I64 71\nEQ\nASSERT\n'
            body+='PUSH_VOID\nSTORE_LOCAL 0\nLOAD_LOCAL 1\nARR_POP\nPUSH_VOID\nSTORE_LOCAL 1\nPUSH_STR a\nEQ\nASSERT\n'
        self.paired(body)

    def test_slice_endpoint_fallback_wrap_and_calls(self):
        body='PUSH_I64 10\nPUSH_I64 20\nPUSH_I64 30\nARR_LITERAL 1 3\nSTORE_LOCAL 0\n'
        for start,end,length in [('PUSH_I64 0','PUSH_I64 3',3),('PUSH_I64 -1','PUSH_I64 3',0),
                                 ('PUSH_I64 2','PUSH_I64 1',0),('PUSH_I64 4294967297','PUSH_I64 -1',2),
                                 ('PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT','PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT',3)]:
            body+=f'LOAD_LOCAL 0\n{start}\n{end}\nARR_SLICE\nARR_LEN\nPUSH_I64 {length}\nEQ\nASSERT\n'
        body+='LOAD_LOCAL 0\nCALL copy\nPUSH_I64 0\nARR_GET\nPUSH_I64 20\nEQ\nASSERT\n'
        suffix='.function copy 1 1 0 array 1\n.parameters copy array\nLOAD_LOCAL 0\nPUSH_I64 1\nPUSH_I64 3\nARR_SLICE\nRET\n.end\n'
        self.paired(body,suffix)

    def test_prepared_split_copy_and_mixed_origin_join(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nPUSH_VOID\nPUSH_VOID\nARR_SLICE\nPUSH_I64 1\nPUSH_I64 42\nARR_SET\nPUSH_I64 1\nARR_GET\nPUSH_I64 42\nEQ\nASSERT\n'
        body+='PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 8\nARR_LITERAL 1 1\nJMP joined\nother:\nPUSH_U8 8\nARR_LITERAL 2 1\njoined:\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nPUSH_I64 9\nARR_PUSH\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT\n'
        self.paired(body)

    def test_counted_owner_allocation_failure_and_recovery(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_LOCAL 0\n'+'LOAD_LOCAL 0\n'*17+'ARR_LITERAL 5 17\nPUSH_VOID\nSTORE_LOCAL 0\nPUSH_I64 0\nPUSH_I64 17\nARR_SLICE\nPOP\n'
        _,ir,wasm=self.compile(self.program(body))
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        self.native_harness(ir,'int failures=0,successes=0;for(int b=0;b<12;b++){budget=b;uint64_t s=nano_try_entry();if(s==((uint64_t)3<<32))failures++;else if(!s)successes++;else return 1;if(nms_module_live_objects())return 2;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 3;}if(!failures||!successes)return 4;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        self.node(wasm,'for(let i=0;i<50;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')

    def test_slice_type_failure_cleans_heap_bounds_and_keeps_global(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nDUP\nDUP\nARR_SLICE\nPOP\n'
        _,ir,wasm=self.compile(self.program(body),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<3;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<3;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_copy_globals_reentry_and_initializer(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE existing\nPUSH_I64 10\nPUSH_I64 20\nARR_LITERAL 1 2\nSTORE_GLOBAL 0\n'
              'existing:\nLOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 -1\nARR_SLICE\nSTORE_LOCAL 0\n'
              'LOAD_GLOBAL 0\nPUSH_I64 30\nARR_PUSH\nPOP\nLOAD_GLOBAL 1\nPUSH_I64 99\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nARR_LEN\nRET\n')
        init='.function __init__ 0 0 0 void 0\nPUSH_I64 99\nARR_LITERAL 1 1\nARR_POP\nSTORE_GLOBAL 1\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program(body,init,result=''),vm_ok=False)
        self.native_harness(ir,'for(int i=2;i<20;i++)if(nano_try_entry()!=(uint64_t)i||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=2;i<20;i++)check(e.nano_try_entry()===BigInt(i));check(e.nms_module_live_objects()===1n);check(e.nano_dispose()===0);e=new WebAssembly.Instance(m).exports;check(e.nano_try_entry()===2n);check(e.nano_dispose()===0);')

    def test_wasm_copy_exhaustion_cleans_frame_and_preserves_roots(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE copy\nARR_LITERAL 1 0\nSTORE_GLOBAL 0\n'
              'PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nLOAD_GLOBAL 0\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 0\nPUSH_I64 32000\nLT\nJMP_TRUE loop\n'
              'copy:\nLOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 32000\nARR_SLICE\nSTORE_GLOBAL 1\n'
              'LOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 32000\nARR_SLICE\nSTORE_GLOBAL 2\n'
              'LOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 32000\nARR_SLICE\nPOP\n')
        _,ir,wasm=self.compile(self.program(body))
        self.native_harness(ir,'for(int i=0;i<3;i++)if(nano_try_entry()||nms_module_live_objects()!=3)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<3;i++){check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===3n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_refusal_preserves_outputs(self):
        for body in ('ARR_NEW 1\nARR_LITERAL 5 1\nPOP\n','PUSH_STR a\nARR_LITERAL 1 1\nPOP\n',
                     'ARR_NEW 7\nPUSH_I64 0\nPUSH_I64 1\nARR_SLICE\nPOP\n'):
            assembly,module=self.work/'bad.nasm',self.work/'bad.nvm'
            assembly.write_text(self.program(body));self.run_cmd([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                output=self.work/'prior';output.write_bytes(b'previous output')
                self.run_cmd([ROOT/'bin'/tool,module,'-o',output],success=False)
                self.assertEqual(output.read_bytes(),b'previous output')

if __name__=='__main__':unittest.main()
