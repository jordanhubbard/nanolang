"""I qualify generated graph ownership through normal VM and target admission."""
import unittest
import re
from tests import test_llvm_managed_mutable_arrays as arrays
ROOT=arrays.ROOT
class ManagedGraphs(unittest.TestCase):
    setUp=arrays.MutableArrays.setUp
    run_cmd=arrays.MutableArrays.run_cmd
    program=arrays.MutableArrays.program
    compile=arrays.MutableArrays.compile
    native_harness=arrays.MutableArrays.native_harness
    node=arrays.MutableArrays.node
    paired=arrays.MutableArrays.paired

    def test_nested_literal_slice_get_pop_and_calls(self):
        body=('ARR_NEW 5\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nLOAD_LOCAL 0\nARR_LITERAL 7 2\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_I64 2\nARR_SLICE\nSTORE_LOCAL 2\n'
              'LOAD_LOCAL 1\nLOAD_LOCAL 2\nEQ\nNOT\nASSERT\n'
              'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nPUSH_STR a\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_STR a\nEQ\nASSERT\n'
              'LOAD_LOCAL 1\nCALL take\nLOAD_LOCAL 0\nEQ\nASSERT\n'
              'LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n')
        suffix=('.function take 1 1 0 array 1\n.parameters take array\n'
                'LOAD_LOCAL 0\nARR_POP\nRET\n.end\n')
        body+=('ARR_NEW 7\nARR_NEW 5\nARR_PUSH\nCALL take\n'
               'PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nARR_PUSH\n'
               'PUSH_I64 0\nARR_GET\nPUSH_STR a\nEQ\nASSERT\n')
        self.paired(body,suffix)

    def test_leaf_mode_stays_unprepared(self):
        for body,graph in [('ARR_NEW 5\nPUSH_STR a\nARR_PUSH\nPOP\n',False),
                           ('ARR_NEW 7\nDUP\nARR_PUSH\nPOP\n',True)]:
            _,ir,_=self.compile(self.program(body))
            text=ir.read_text()
            self.assertEqual('gc_status = call i32 @nms_module_graph_collect()' in text,graph)
            entry=text.split('define i64 @nano_try_entry()')[1]
            self.assertEqual('call i32 @nms_module_graph_begin' in entry,graph)
            self.assertEqual('prepare_failed:' in entry,graph)

    def test_prepared_split_nested_append_and_set(self):
        body=('ARR_NEW 7\nPOP\n'
              'PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nARR_NEW 1\nARR_PUSH\nARR_POP\nARR_LEN\nPUSH_I64 0\nEQ\nASSERT\n'
              'PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nPUSH_I64 0\nARR_NEW 1\nARR_SET\n'
              'PUSH_I64 0\nARR_GET\nPUSH_I64 12\nARR_PUSH\nARR_POP\nPUSH_I64 12\nEQ\nASSERT\n')
        self.paired(body)

    def test_self_mutual_cycles_and_bounded_live_loop(self):
        body=('PUSH_I64 0\nSTORE_LOCAL 3\nloop:\n'
              'ARR_NEW 7\nSTORE_LOCAL 0\nARR_NEW 5\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 0\nLOAD_LOCAL 1\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 1\nPUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 1\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 3\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 3\n'
              'PUSH_I64 12000\nLT\nJMP_TRUE loop\n')
        self.paired(body)

    def test_global_cycles_reentry_initializer_and_first_error(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE exists\n'
              'ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_GLOBAL 0\n'
              'exists:\nLOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nLOAD_GLOBAL 0\nEQ\nASSERT\n'
              'LOAD_GLOBAL 0\nPUSH_I64 9\nARR_PUSH\nPOP\n'
              'LOAD_GLOBAL 1\nPUSH_I64 123\nEQ\nASSERT\n'
              'ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_LOCAL 0\nPUSH_BOOL 0\nASSERT\n')
        suffix='.function __init__ 0 0 0 void 0\nPUSH_I64 123\nSTORE_GLOBAL 1\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program(body,suffix),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<12;i++)if(nano_try_entry()!=((uint64_t)2<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<12;i++){check(e.nano_try_entry()===(2n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_preparation_growth_failures_and_nested_entry(self):
        body=('ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_LOCAL 0\n'
              'PUSH_I64 0\nSTORE_LOCAL 3\nfill:\nLOAD_LOCAL 0\nARR_NEW 7\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 3\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 3\nPUSH_I64 20\nLT\nJMP_TRUE fill\n'
              +'LOAD_LOCAL 0\n'*17+'ARR_LITERAL 7 17\nPUSH_I64 0\nPUSH_I64 17\nARR_SLICE\nPOP\n')
        _,ir,wasm=self.compile(self.program(body))
        # I redirect only this generated module, before its existing ASan pass.
        # The separate harness and sanitizer runtime keep normal allocations.
        original_ir=ir.read_text()
        self.assertRegex(original_ir,r'declare[^\n]*@malloc\(')
        self.assertRegex(original_ir,r'call[^\n]*@malloc\(')
        controlled_ir=self.work/'allocation-controlled.ll'
        controlled_ir.write_text(re.sub(r'@malloc(?=\()','@nano_test_graph_malloc',original_ir))
        extra=('static long budget=-1;'
               'void *nano_test_graph_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}')
        # Fresh executables independently cover begin allocation and subsequent
        # table/workspace/buffer preparation; every failure must end active entry.
        for b in (0,1,2,3,4,6,10,15,25,45,80):
            self.native_harness(controlled_ir,f'budget={b};uint64_t s=nano_try_entry();if(s && s!=((uint64_t)3<<32))return 1;if({b}==0 && s!=((uint64_t)3<<32))return 8;if(nms_module_live_objects())return 2;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 3;return nano_dispose();',extra)
        self.assertEqual(ir.read_text(),original_ir)
        extra=('extern unsigned nms_module_graph_begin(const void*,unsigned),nms_module_active(void),nms_module_status(void);'
               'extern void nms_module_fail(unsigned);extern uint64_t nms_module_graph_finish(int);')
        self.native_harness(ir,'if(nms_module_graph_begin(0,0))return 1;nms_module_fail(2);if(nano_try_entry()!=((uint64_t)4<<32)||!nms_module_active()||nms_module_status()!=2)return 2;if(nms_module_graph_finish(0)!=((uint64_t)2<<32))return 3;if(nano_try_entry())return 4;if(nano_dispose())return 5;return nano_try_entry()!=((uint64_t)5<<32);',extra)
        self.node(wasm,'for(let i=0;i<10;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')

    def test_generated_pressure_preserves_live_graph_and_disposal(self):
        body=('ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_GLOBAL 0\n'
              'ARR_NEW 1\nSTORE_LOCAL 0\nLOAD_GLOBAL 0\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
              'PUSH_I64 0\nSTORE_LOCAL 3\nfill:\nLOAD_LOCAL 0\nLOAD_LOCAL 3\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 3\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 3\nPUSH_I64 200000\nLT\nJMP_TRUE fill\n'
              'PUSH_VOID\nSTORE_GLOBAL 0\n')
        _,ir,wasm=self.compile(self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<3;i++){check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===2n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);e=new WebAssembly.Instance(m).exports;check(e.nano_try_entry()===(3n<<32n));check(e.nano_dispose()===0);')

    def test_initializer_and_helper_failures_collect_only_dead_roots(self):
        cycle='ARR_NEW 7\nDUP\nARR_PUSH\nSTORE_LOCAL 0\n'
        for bad,status in [('LOAD_LOCAL 0\nPUSH_I64 -1\nLOAD_LOCAL 0\nARR_SET\nPOP\n',7),
                           ('LOAD_LOCAL 0\nLOAD_LOCAL 0\nARR_GET\nPOP\n',1)]:
            suffix='.function fail 1 1 0 void 0\n.parameters fail array\n'+bad+'RET\n.end\n'
            _,ir,wasm=self.compile(self.program(cycle+'LOAD_LOCAL 0\nCALL fail\n',suffix),vm_ok=False)
            self.native_harness(ir,f'for(int i=0;i<3;i++)if(nano_try_entry()!=((uint64_t){status}<<32)||nms_module_live_objects())return 1;return nano_dispose();')
            self.node(wasm,f'for(let i=0;i<3;i++){{check(e.nano_try_entry()===({status}n<<32n));check(e.nms_module_live_objects()===0n);}}check(e.nano_dispose()===0);')
        init='.function __init__ 0 1 0 void 0\n'+cycle+'PUSH_BOOL 0\nASSERT\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program('',init),vm_ok=False)
        self.native_harness(ir,'if(nano_try_entry()!=((uint64_t)2<<32)||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,'check(e.nano_try_entry()===(2n<<32n));check(e.nms_module_live_objects()===0n);check(e.nano_dispose()===0);')

if __name__=='__main__':unittest.main()
