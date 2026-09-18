"""I qualify checked ordinary record ownership on VM, native LLVM and Wasm."""
import os
import shlex
import shutil
from pathlib import Path
import unittest
from tests import test_llvm_managed_strings as managed
from tests.test_managed_record_shapes import program, function, NO
ROOT = managed.ROOT

class ManagedRecords(unittest.TestCase):
    def setUp(self):
        managed.ManagedStrings.setUp(self)
        if os.environ.get('NMR_ARTIFACTS'):
            destination=Path(os.environ['NMR_ARTIFACTS'])/self.id()
            self.addCleanup(lambda: shutil.copytree(self.work,destination,dirs_exist_ok=True))
    run_cmd = managed.ManagedStrings.run_cmd
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def paired(self, text):
        module, ir, wasm = self.compile(text)
        self.assertIn('call i64 @nms_module_record_begin', ir.read_text())
        self.native_harness(ir, 'for(int i=0;i<3;i++)if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm, 'for(let i=0;i<3;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')
        public=self.work/'public.wasm'
        self.run_cmd([ROOT/'bin/nvm2wasm',module,'-o',public])
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',public]).stdout,'0\n')
        return ir, wasm

    def test_exact_scalar_payloads_counted_constructors_and_empty_identity(self):
        layouts=[(0,[]),(3,[]),(0,[(1,NO),(2,NO),(3,NO),(4,NO),(5,NO)]),(0,[(1,NO)])]
        body=('STRUCT_NEW 0\nDUP\nEQ\nASSERT\nSTRUCT_NEW 0\nSTRUCT_NEW 0\nNE\nASSERT\n')
        for bits in [-9223372036854775808,1,9218868437227405311,9221120237041090645,-2251799813685227]:
            for ctor in ['STRUCT_LITERAL 1 5','AGG_PACK 0 1 0 5']:
                body+=('PUSH_I64 -9223372036854775808\nPUSH_U8 255\n'
                       f'PUSH_I64 {bits}\nF64_FROM_BITS\nPUSH_BOOL 1\nPUSH_STR text\n{ctor}\nSTORE_LOCAL 0\n'
                       'LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 -9223372036854775808\nEQ\nASSERT\n'
                       'LOAD_LOCAL 0\nSTRUCT_GET 1\nPUSH_U8 255\nEQ\nASSERT\n'
                       f'LOAD_LOCAL 0\nAGG_GET 2\nF64_TO_BITS\nPUSH_I64 {bits}\nEQ\nASSERT\n'
                       'LOAD_LOCAL 0\nSTRUCT_GET 3\nASSERT\nLOAD_LOCAL 0\nAGG_GET 4\nPUSH_STR text\nEQ\nASSERT\n')
        body+='PUSH_I64 9\nSTRUCT_LITERAL 2 1\nAGG_GET 0\nPUSH_I64 9\nEQ\nASSERT\n'
        self.paired(program(body,layouts,[8]))
        self.paired(program('PUSH_I64 3\n'*256+'STRUCT_LITERAL 0 256\nAGG_GET 255\nPUSH_I64 3\nEQ\nASSERT',[(0,[(1,NO)]*256)]))

    def test_nested_aliases_get_temporary_set_and_reordered_calls(self):
        layouts=[(0,[(5,NO)]),(0,[(8,0),(8,0)])]
        body=('PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 0 1\nDUP\nSTRUCT_LITERAL 1 2\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 0\nAGG_GET 0\nLOAD_LOCAL 0\nAGG_GET 1\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nAGG_GET 0\nPUSH_STR empty\nAGG_SET 0\nPOP\n'
              'LOAD_LOCAL 0\nAGG_GET 1\nAGG_GET 0\nPUSH_STR empty\nEQ\nASSERT\n'
              'PUSH_STR text\nSTRUCT_LITERAL 0 1\nLOAD_LOCAL 0\nAGG_GET 0\nCALL second\nAGG_GET 0\nPUSH_STR empty\nEQ\nASSERT\n'
              'LOAD_LOCAL 0\nAGG_GET 0\nLOAD_LOCAL 0\nAGG_GET 1\nAGG_GET 0\nSTRUCT_SET 0\nPOP\n'
              'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nPUSH_STR empty\nEQ\nASSERT\n')
        helper=function('second','LOAD_LOCAL 1\nRET',[8,8],2,8)
        self.paired(program(body,layouts,[8],helpers=[helper]))
        self.paired(program('PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 0 1\nSTRUCT_GET 0\nPUSH_STR text\nEQ\nASSERT',[(0,[(5,NO)])]))

    def test_generic_consumers_identity_casts_order_and_branches(self):
        body='STRUCT_NEW 0\nSTORE_LOCAL 0\n'
        for op in ('EQ','LE','GE'):
            body+=f'LOAD_LOCAL 0\nLOAD_LOCAL 0\n{op}\nASSERT\n'
        for op in ('LT','GT'):
            body+=f'LOAD_LOCAL 0\nSTRUCT_NEW 0\n{op}\nNOT\nASSERT\n'
        body+=('LOAD_LOCAL 0\nTYPE_CHECK 8\nASSERT\nLOAD_LOCAL 0\nASSERT\n'
               'LOAD_LOCAL 0\nCAST_INT\nPUSH_I64 0\nEQ\nASSERT\n'
               'LOAD_LOCAL 0\nCAST_FLOAT\nF64_TO_BITS\nPUSH_I64 0\nEQ\nASSERT\n'
               'LOAD_LOCAL 0\nCAST_STRING\nPUSH_STR empty\nEQ\nASSERT\n'
               'LOAD_LOCAL 0\nPUSH_I64 0\nGT\nASSERT\n'
               'PUSH_BOOL 1\nJMP_FALSE other\nLOAD_LOCAL 0\nJMP joined\nother:\nSTRUCT_NEW 0\njoined:\nPOP\n')
        self.paired(program(body,[(0,[])],[8]))

    def test_weak_site_churn_and_independent_array_cycles(self):
        body=('PUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
              'PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 0 1\nSTORE_LOCAL 1\n'
              'LOAD_LOCAL 1\nSTRUCT_LITERAL 1 1\nAGG_GET 0\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n'
              'ARR_NEW 7\nDUP\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 0\nPUSH_I64 16000\nLT\nJMP_TRUE loop\n')
        self.paired(program(body,[(0,[(5,NO)]),(0,[(8,0)])],[1,8]))

    def test_global_identity_initializer_reentry_and_failure_cleanup(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE exists\n'
              'PUSH_I64 0\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\nexists:\n'
              'LOAD_GLOBAL 1\nPUSH_I64 17\nEQ\nASSERT\n'
              'LOAD_GLOBAL 0\nDUP\nAGG_GET 0\nPUSH_I64 1\nI64_ADD\nSTRUCT_SET 0\nLOAD_GLOBAL 0\nEQ\nASSERT\n'
              'PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 1 1\nPOP\nPUSH_BOOL 0\nASSERT\n')
        init=function('__init__','PUSH_I64 17\nSTORE_GLOBAL 1\nRET',[],0,0)
        _,ir,wasm=self.compile(program(body,[(0,[(1,NO)]),(0,[(5,NO)])],helpers=[init]),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<25;i++)if(nano_try_entry()!=((uint64_t)2<<32)||nms_module_live_objects()!=1)return 1;if(nano_dispose()||nms_module_live_objects())return 2;return nano_try_entry()!=((uint64_t)5<<32);')
        self.node(wasm,'for(let i=0;i<25;i++){check(e.nano_try_entry()===(2n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);check(e.nano_try_entry()===(5n<<32n));e=new WebAssembly.Instance(m).exports;check(e.nano_try_entry()===(2n<<32n));check(e.nano_dispose()===0);')

    def test_allocation_pressure_acquired_failure_and_nested_begin(self):
        body=('PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 0 1\nSTORE_LOCAL 0\n'
              +'LOAD_LOCAL 0\n'*20+'STRUCT_LITERAL 1 20\nPOP\n')
        _,ir,wasm=self.compile(program(body,[(0,[(5,NO)]),(0,[(8,0)]*20)],[8]))
        extra='static long budget=-1;void *nano_test_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}'
        for budget in (0,1,2,3,4,5,6,7,8,12):
            self.native_harness(ir,f'budget={budget};uint64_t s=nano_try_entry();if(s && s!=((uint64_t)3<<32))return 1;if({budget}==0 && s!=((uint64_t)3<<32))return 2;if(nms_module_live_objects())return 3;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 4;return nano_dispose();',extra,allocation_control=True)
        extra=('extern unsigned nms_module_graph_begin(const void*,unsigned),nms_module_active(void),nms_module_status(void);'
               'extern void nms_module_fail(unsigned);extern uint64_t nms_module_graph_finish(int);')
        self.native_harness(ir,'if(nms_module_graph_begin(0,0))return 1;nms_module_fail(2);if(nano_try_entry()!=((uint64_t)4<<32)||!nms_module_active()||nms_module_status()!=2)return 2;if(nms_module_graph_finish(0)!=((uint64_t)2<<32))return 3;return nano_dispose();',extra)
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        self.native_harness(ir,'if(nano_dispose())return 1;return nano_try_entry()!=((uint64_t)5<<32);')

    def test_receiver_error_families_and_counted_failure_inputs(self):
        for op,status in [('STRUCT_GET',1),('AGG_GET',7),('STRUCT_SET',1),('AGG_SET',7)]:
            value='PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\n' if op.endswith('SET') else ''
            body=('PUSH_STR text\nPUSH_STR empty\nSTR_CONCAT\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\n'
                  'PUSH_BOOL 1\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\n'+value+op+' 0\nPOP\n')
            _,ir,wasm=self.compile(program(body,[(0,[(5,NO)])]),vm_ok=False)
            self.native_harness(ir,f'for(int i=0;i<3;i++)if(nano_try_entry()!=((uint64_t){status}<<32)||nms_module_live_objects()!=2)return 1;if(nano_dispose())return 2;return nms_module_live_objects()!=0;')
            self.node(wasm,f'for(let i=0;i<3;i++){{check(e.nano_try_entry()===({status}n<<32n));check(e.nms_module_live_objects()===2n);}}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_committed_global_survives_allocation_failure_reentry(self):
        body=('LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE existing\nPUSH_I64 42\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\n'
              'existing:\nLOAD_GLOBAL 0\nAGG_GET 0\nPUSH_I64 42\nEQ\nASSERT\n'
              'PUSH_I64 19\nSTRUCT_LITERAL 0 1\nPOP\n'
              'LOAD_GLOBAL 0\nPUSH_I64 42\nSTRUCT_SET 0\nLOAD_GLOBAL 0\nEQ\nASSERT\n')
        _,ir,wasm=self.compile(program(body))
        extra='static long budget=-1;void *nano_test_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}'
        for budget in (0,1,2,3):
            self.native_harness(ir,f'if(nano_try_entry()||nms_module_live_objects()!=1)return 1;budget={budget};uint64_t s=nano_try_entry();if(s && s!=((uint64_t)3<<32))return 2;if({budget}==0 && s!=((uint64_t)3<<32))return 3;if(nms_module_live_objects()!=1)return 4;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 5;return nano_dispose();',extra,allocation_control=True)
        self.node(wasm,'for(let i=0;i<20;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_unknown_wrong_nominal_and_mixed_edges_preserve_output(self):
        cases=[program('PUSH_I64 1\nSTRUCT_LITERAL 0 1\nPOP',authority=False),
               program('STRUCT_NEW 0\nPOP'),
               program('PUSH_I64 1\nSTRUCT_LITERAL 1 1\nSTRUCT_LITERAL 2 1\nPOP',[(0,[(1,NO)]),(0,[(1,NO)]),(0,[(8,0)])]),
               program('ARR_NEW 1\nSTRUCT_LITERAL 0 1\nPOP',[(0,[(7,NO)])],authority=False),
               program('PUSH_I64 1\nSTRUCT_LITERAL 0 1\nARR_LITERAL 8 1\nPOP')]
        for text in cases:
            source,module=self.work/'refuse.nasm',self.work/'refuse.nvm'
            source.write_text(text);self.run_cmd([ROOT/'bin/nanoisa','asm',source,'-o',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                output=self.work/'prior';output.write_bytes(b'previous output')
                self.run_cmd([ROOT/'bin'/tool,module,'-o',output],success=False)
                self.assertEqual(output.read_bytes(),b'previous output')

class HeapSelector(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    def test_selector_atomicity(self):
        objects=shlex.split(os.environ['NMA_LINK_OBJECTS'])
        source=self.work/'selector.nasm'
        source.write_text(program('PUSH_STR text\nSTRUCT_LITERAL 0 1\nPOP',[(0,[(5,NO)])]))
        for compiler,flags in [('cc',[]),('clang',shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+['-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            probe=self.work/'selector'
            self.run_cmd([compiler,*flags,'-std=c11','-O1','-Wall','-Wextra','-Werror','-DNMA_TESTING',
                ROOT/'src/nanoisa/managed_array_shapes.c',ROOT/'tests/nanoisa/test_managed_heap_selector.c',*objects,'-lm','-lcrypto','-o',probe])
            self.run_cmd([probe,source])

if __name__=='__main__':unittest.main()
