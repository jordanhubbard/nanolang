"""I qualify shape-checked mutable arrays on VM, native LLVM and import-free Wasm."""
import unittest
from tests import test_llvm_managed_strings as managed
ROOT = managed.ROOT

class MutableArrays(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    program = managed.ManagedStrings.program
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def paired(self, body, suffix='', prefix=''):
        module, ir, wasm = self.compile(prefix+self.program(body,suffix))
        self.native_harness(ir, 'for(int i=0;i<3;i++)if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm, 'for(let i=0;i<3;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')
        published=self.work/'published.wasm'
        self.run_cmd([ROOT/'bin/nvm2wasm',module,'-o',published])
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',published]).stdout,'0\n')

    def test_packed_matrix_optional_results_and_growth(self):
        cases=[(1,'PUSH_I64 -9223372036854775808','PUSH_I64 -9223372036854775808'),
               (1,'PUSH_U8 255','PUSH_I64 255'),(2,'PUSH_I64 -257','PUSH_U8 255'),
               (2,'PUSH_U8 255','PUSH_U8 255'),(3,'PUSH_I64 9007199254740993','PUSH_F64 9007199254740992'),
               (3,'PUSH_F64 -0.0','PUSH_F64 -0.0'),(3,'PUSH_F64 2.2250738585072014e-308\nPUSH_F64 4503599627370496\nF64_DIV','PUSH_F64 2.2250738585072014e-308\nPUSH_F64 4503599627370496\nF64_DIV'),
               (3,'PUSH_F64 inf','PUSH_F64 inf'),(4,'PUSH_BOOL 1','PUSH_BOOL 1')]
        body=''
        for kind,value,expected in cases:
            body+=f'ARR_NEW {kind}\nSTORE_LOCAL 0\n'
            for _ in range(33): body+=f'LOAD_LOCAL 0\n{value}\nARR_PUSH\nPOP\n'
            body+='LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 33\nEQ\nASSERT\n'
            body+=f'LOAD_LOCAL 0\nPUSH_I64 0\n{value}\nARR_SET\nLOAD_LOCAL 0\nEQ\nASSERT\n'
            for idx in (0,32):body+=f'LOAD_LOCAL 0\nPUSH_I64 {idx}\nARR_GET\nDUP\nTYPE_CHECK {kind}\nASSERT\n{expected}\nEQ\nASSERT\n'
            for idx in (-1,33,4294967296):body+=f'LOAD_LOCAL 0\nPUSH_I64 {idx}\nARR_GET\nTYPE_CHECK 0\nASSERT\n'
            for _ in range(33):body+=f'LOAD_LOCAL 0\nARR_POP\n{expected}\nEQ\nASSERT\n'
            body+='LOAD_LOCAL 0\nARR_POP\nTYPE_CHECK 0\nASSERT\nPUSH_VOID\nSTORE_LOCAL 0\n'
        self.paired(body)

    def test_boxed_leaf_aliases_calls_and_split_mutation(self):
        body='ARR_NEW 5\nDUP\nSTORE_GLOBAL 0\nSTORE_LOCAL 0\n'
        values=[('PUSH_VOID',0),('PUSH_I64 -9',1),('PUSH_U8 255',2),('PUSH_F64 0.5',3),('PUSH_BOOL 1',4),('PUSH_STR a',5)]
        for value,tag in values:
            body+=f'LOAD_LOCAL 0\n{value}\nARR_PUSH\nPOP\nLOAD_GLOBAL 0\nARR_POP\nDUP\nTYPE_CHECK {tag}\nASSERT\n{value}\nEQ\nASSERT\n'
        body+='LOAD_LOCAL 0\nCALL write\nLOAD_GLOBAL 0\nEQ\nASSERT\n'
        body+='LOAD_LOCAL 0\nPUSH_I64 0\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nARR_SET\nPOP\n'
        body+='LOAD_LOCAL 0\nARR_POP\nPUSH_VOID\nSTORE_LOCAL 0\nPUSH_VOID\nSTORE_GLOBAL 0\nPUSH_STR a\nEQ\nASSERT\n'
        body+='PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nDUP\nSTORE_LOCAL 0\nPUSH_I64 1\nPUSH_I64 42\nARR_SET\nLOAD_LOCAL 0\nEQ\nASSERT\n'
        body+='LOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nPUSH_I64 42\nEQ\nASSERT\nPUSH_VOID\nSTORE_LOCAL 0\n'
        suffix='.function write 1 1 0 array 1\n.parameters write array\nLOAD_LOCAL 0\nPUSH_STR a\nARR_PUSH\nRET\n.end\n'
        self.paired(body,suffix)

    def test_checked_errors_release_frames_and_preserve_alias_writes(self):
        for bad,status in [('LOAD_LOCAL 0\nPUSH_I64 -1\nPUSH_STR a\nARR_SET\nPOP\n',7),
                           ('LOAD_LOCAL 0\nPUSH_I64 4294967296\nPUSH_STR a\nARR_SET\nPOP\n',7),
                           ('LOAD_LOCAL 0\nLOAD_GLOBAL 1\nPUSH_STR a\nARR_SET\nPOP\n',1),
                           ('LOAD_GLOBAL 1\nPUSH_STR a\nARR_PUSH\nPOP\n',1),
                           ('LOAD_GLOBAL 1\nARR_POP\nPOP\n',1)]:
            with self.subTest(bad=bad):
                body='ARR_NEW 5\nPUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nARR_PUSH\nDUP\nSTORE_GLOBAL 0\nPUSH_BOOL 1\nSTORE_GLOBAL 1\nCALL fail\nPOP\n'
                suffix='.function fail 1 1 0 array 1\n.parameters fail array\n'+bad+'LOAD_LOCAL 0\nRET\n.end\n'
                _,ir,wasm=self.compile(self.program(body,suffix),vm_ok=False)
                self.native_harness(ir,f'for(int i=0;i<3;i++)if(nano_try_entry()!=((uint64_t){status}<<32)||nms_module_live_objects()!=2)return 1;return nano_dispose();')
                self.node(wasm,f'for(let i=0;i<3;i++){{check(e.nano_try_entry()===({status}n<<32n));check(e.nms_module_live_objects()===2n);}}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_reentry_global_and_initializer_order(self):
        body='LOAD_GLOBAL 0\nTYPE_CHECK 0\nJMP_FALSE exists\nARR_NEW 1\nSTORE_GLOBAL 0\nexists:\nLOAD_GLOBAL 0\nPUSH_I64 7\nARR_PUSH\nPOP\nLOAD_GLOBAL 1\nPUSH_I64 9\nEQ\nASSERT\nLOAD_GLOBAL 0\nARR_LEN\nRET\n'
        init='.function __init__ 0 0 0 void 0\nPUSH_I64 9\nSTORE_GLOBAL 1\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program(body,init,result=''),vm_ok=False)
        self.native_harness(ir,'for(int i=1;i<40;i++)if(nano_try_entry()!=(uint64_t)i||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=1;i<40;i++)check(e.nano_try_entry()===BigInt(i));check(e.nano_dispose()===0);e=new WebAssembly.Instance(m).exports;check(e.nano_try_entry()===1n);check(e.nano_dispose()===0);')

    def test_allocation_failures_cleanup_and_reuse(self):
        body=('ARR_NEW 5\nSTORE_GLOBAL 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
              'LOAD_GLOBAL 0\nPUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 0\nPUSH_I64 40\nLT\nJMP_TRUE loop\n'
              'PUSH_VOID\nSTORE_GLOBAL 0\n')
        _,ir,_=self.compile(self.program(body))
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        # Every failed module entry clears its frame, retaining only committed
        # global roots; a corrected subsequent entry then releases those roots.
        for budget in (0,1,2,3,9,18,35):
            self.native_harness(ir,f'budget={budget};if(nano_try_entry()!=((uint64_t)3<<32))return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        body=('ARR_NEW 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
              'LOAD_GLOBAL 0\nLOAD_LOCAL 0\nARR_PUSH\nPOP\n'
              'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 0\nPUSH_I64 200000\nLT\nJMP_TRUE loop\n'
              'PUSH_VOID\nSTORE_GLOBAL 0\n')
        _,ir,wasm=self.compile(self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<3;i++){check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_unsupported_shapes_and_transfers_preserve_output(self):
        for body in ['ARR_NEW 5\nARR_NEW 1\nARR_PUSH\nPOP\n',
                     'ARR_NEW 1\nPUSH_STR a\nARR_PUSH\nPOP\n',
                     'ARR_NEW 1\nPOP\nPUSH_I64 1\nPUSH_I64 2\nADD\nPOP\n']:
            assembly,module=self.work/'refuse.nasm',self.work/'refuse.nvm'
            assembly.write_text(self.program(body));self.run_cmd([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
            for tool in ('nvm2llvm','nvm2wasm'):
                output=self.work/'prior';output.write_bytes(b'previous output')
                result=self.run_cmd([ROOT/'bin'/tool,module,'-o',output],success=False)
                self.assertIn('mutable array eligibility',result.stderr)
                self.assertEqual(output.read_bytes(),b'previous output')

if __name__=='__main__':unittest.main()
