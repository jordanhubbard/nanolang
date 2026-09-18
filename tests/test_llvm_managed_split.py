"""I qualify owned split arrays, exact bytes and read-only access on real targets."""
import unittest
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes
ROOT = managed.ROOT
CASES = [(b'', b''), (b'', b','), (b'abc', b''), (b',a,,b,', b','),
         (b'a\0b\0', b'\0'), (b'aaaaa', b'aa'), (b'abc', b'xyz'),
         (b'\xff\x80\0', b''), (b'left--right--', b'--')]

class ManagedSplit(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    program = managed.ManagedStrings.program
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def test_bytes_aliases_calls_globals_missing_and_scalar_interactions(self):
        strings = ''; body = ''
        for i, (source, delim) in enumerate(CASES):
            parts = source.split(delim) if delim else [source[k:k+1] for k in range(len(source))]
            strings += f'.string s{i} {c_bytes(source)}\n.string d{i} {c_bytes(delim)}\n'
            body += f'PUSH_STR s{i}\nPUSH_STR empty\nSTR_CONCAT\nPUSH_STR d{i}\nCALL split\nDUP\nTYPE_CHECK 7\nASSERT\nDUP\nSTORE_GLOBAL 0\nCALL relay\nSTORE_LOCAL 0\n'
            body += f'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 {len(parts)}\nEQ\nASSERT\n'
            body += 'LOAD_LOCAL 0\nLOAD_GLOBAL 0\nEQ\nASSERT\nLOAD_LOCAL 0\nCAST_BOOL\nASSERT\n'
            body += 'LOAD_LOCAL 0\nCAST_INT\nPUSH_I64 0\nEQ\nASSERT\nLOAD_LOCAL 0\nCAST_FLOAT\nPUSH_F64 0\nEQ\nASSERT\n'
            body += 'LOAD_LOCAL 0\nCAST_STRING\nPUSH_STR empty\nEQ\nASSERT\n'
            body += f'PUSH_STR s{i}\nPUSH_STR d{i}\nSTR_SPLIT\nDUP\nLOAD_LOCAL 0\nNE\nASSERT\nLOAD_LOCAL 0\nLE\nASSERT\n'
            for k, part in enumerate(parts):
                strings += f'.string p{i}_{k} {c_bytes(part)}\n'
                body += f'LOAD_LOCAL 0\nPUSH_I64 {k}\nARR_GET\nPUSH_STR p{i}_{k}\nEQ\nASSERT\n'
            for index in (-1, len(parts), 4294967296, 9223372036854775807):
                body += f'LOAD_LOCAL 0\nPUSH_I64 {index}\nARR_GET\nPUSH_VOID\nEQ\nASSERT\n'
        # Return a retained child after clearing every array root.
        body += 'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 1\nPUSH_VOID\nSTORE_LOCAL 0\nPUSH_VOID\nSTORE_GLOBAL 0\nLOAD_LOCAL 1\nPUSH_STR p8_0\nEQ\nASSERT\n'
        suffix = ('.function split 2 2 0 array 1\n.parameters split string string\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nSTR_SPLIT\nRET\n.end\n'
                  '.function relay 1 1 0 array 1\n.parameters relay array\nLOAD_LOCAL 0\n.end\n')
        _, ir, wasm = self.compile(strings + self.program(body, suffix))
        self.native_harness(ir, 'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm, 'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);}check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout, '0\n')

    def test_persistent_global_child_and_error_cleanup(self):
        body = ('LOAD_GLOBAL 0\nPUSH_VOID\nEQ\nJMP_FALSE retained\n'
                'PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nSTORE_GLOBAL 0\n'
                'PUSH_I64 7\nSTORE_GLOBAL 1\nJMP joined\nretained:\n'
                'LOAD_GLOBAL 0\nARR_LEN\nPUSH_I64 3\nEQ\nASSERT\njoined:\n'
                'LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nSTR_LEN\nPUSH_I64 1\nEQ\nASSERT\n')
        _, ir, wasm = self.compile(self.program(body))
        self.native_harness(ir, 'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=4)return 1;return nano_dispose();')
        self.node(wasm, 'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===4n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        for bad in ('LOAD_GLOBAL 0\nLOAD_GLOBAL 1\nSTR_SPLIT\nPOP\n',
                    'LOAD_GLOBAL 1\nLOAD_GLOBAL 0\nSTR_SPLIT\nPOP\n',
                    'LOAD_GLOBAL 1\nARR_LEN\nPOP\n',
                    'LOAD_GLOBAL 0\nPUSH_BOOL 1\nSTORE_GLOBAL 2\nLOAD_GLOBAL 2\nARR_GET\nPOP\n',
                    'LOAD_GLOBAL 1\nPUSH_I64 0\nARR_GET\nPOP\n'):
            _, ir, wasm = self.compile(self.program(body + bad), vm_ok=False)
            self.native_harness(ir, 'for(int i=0;i<3;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=4)return 1;return nano_dispose();')
            self.node(wasm, 'for(let i=0;i<3;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===4n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

    def test_partial_allocation_core_and_emitted_failure_recovery(self):
        source = self.work / 'core.c'
        source.write_text('''#include "managed_strings.h"
static unsigned char many[100000];
int run(void){NmsRuntime r;nms_init(&r,0,0);NmsHandle held[8],out=99;
 for(unsigned i=0;i<8;i++)if(nms_create(&r,(const unsigned char*)(i?",":",a,,b,"),i?1:6,&held[i]))return 1;
 uint64_t live=r.live_bytes;
 for(unsigned fail=0;fail<12;fail++){
  if(nms_retain(&r,held[0])||nms_retain(&r,held[1]))return 2;
  nms_test_fail_after(&r,fail);out=99;
  NmsStatus s=nms_split_owned(&r,held[0],held[1],&out);
  if(s==NMS_MEMORY){if(out!=99)return 3;}else if(s||nms_release(&r,out))return 4;
  if(r.live_objects!=8||r.live_bytes!=live)return 5;
  for(unsigned i=0;i<8;i++)if(r.slots[(uint32_t)held[i]].references!=1)return 6;
  nms_test_fail_after(&r,UINT64_MAX);
 }
 if(nms_retain(&r,held[0])||nms_retain(&r,held[0])||nms_split_owned(&r,held[0],held[0],&out))return 7;
 uint32_t length=0;if(nms_string_array_length(&r,out,&length)||length!=2||nms_release(&r,out))return 8;
 if(nms_retain(&r,held[0])||nms_retain(&r,held[1]))return 9;
 if(nms_split_owned(&r,held[0],held[1],0)!=NMS_STATE)return 10;
 for(unsigned i=0;i<8;i++)if(nms_release(&r,held[i]))return 11;
 if(r.live_objects||r.live_bytes||nms_dispose(&r)||nms_test_live_allocations())return 12;
#ifdef __wasm32__
 nms_init(&r,0,0);for(unsigned i=0;i<sizeof many;i++)many[i]='a';
 if(nms_create(&r,many,sizeof many,&held[0])||nms_create(&r,(const unsigned char*)"a",1,&held[1]))return 13;
 out=99;if(nms_split_owned(&r,held[0],held[1],&out)!=NMS_MEMORY||out!=99)return 14;
 if(r.live_objects||r.live_bytes||nms_dispose(&r)||nms_test_live_allocations())return 15;
#endif
 return 0;}
#ifndef __wasm32__
int main(void){return run();}
#endif
''')
        inc = '-I'+str(ROOT/'src/nanoisa'); core = ROOT/'src/nanoisa/managed_strings.c'
        exe, wasm = self.work/'core', self.work/'core.wasm'
        self.run_cmd(self.clang+['-DNMS_TESTING','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',inc,core,source,'-o',exe]); self.run_cmd([exe])
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-DNMS_TESTING','-O2','-ffreestanding','-fno-builtin','-nostdlib',inc,core,source,'-Wl,--no-entry','-Wl,--max-memory=1048576','-Wl,--export=run','-o',wasm])
        self.node(wasm, 'check(e.run()===0);check(e.run()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','run',wasm]).stdout, '0\n')
        body = 'PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nPUSH_STR empty\nSTR_SPLIT\nSTORE_GLOBAL 1\n'
        _, ir, _ = self.compile(self.program(body))
        extra = 'static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        for fail in range(2,6):
            self.native_harness(ir, f'budget={fail};if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=5)return 2;return nano_dispose();', extra, ['-Wl,--wrap=malloc'])

    def test_wasm_memory_limit_cleans_partial_array_and_preserves_global(self):
        text = '.string large ' + c_bytes(b'a' * 100000) + '\n'
        body = ('PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nSTORE_GLOBAL 0\n'
                'PUSH_STR large\nPUSH_STR empty\nSTR_SPLIT\nPOP\n')
        _, ir, wasm = self.compile(text + self.program(body))
        self.native_harness(ir, 'if(nano_try_entry()||nms_module_live_objects()!=4)return 1;return nano_dispose();')
        self.node(wasm, 'for(let i=0;i<3;i++){check(e.nano_try_entry()===(3n<<32n));check(e.nms_module_live_objects()===4n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_try_entry',wasm]).stdout, '12884901888\n')

    def test_prior_split_global_and_array_initializer_result(self):
        suffix = ('.function __init__ 0 0 0 array 1\n'
                  'PUSH_STR text\nPUSH_STR text\nSTR_SPLIT\nRET\n.end\n')
        text = '.string text "ordinary"\n' + self.program(
            'PUSH_STR text\nPUSH_STR text\nSTR_SPLIT\nSTORE_GLOBAL 0\n', suffix)
        _, ir, wasm = self.compile(text)
        self.native_harness(ir, 'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=3)return 1;return nano_dispose();')
        self.node(wasm, 'for(let instance=0;instance<2;instance++){e=new WebAssembly.Instance(m).exports;check(e.nms_module_live_objects()===0n);for(let i=0;i<3;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===3n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);}')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout, '0\n')

    def test_mutation_refusals_preserve_output(self):
        for op in ('ARR_NEW 5\nPOP\n', 'PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nPUSH_STR a\nARR_PUSH\nPOP\n',
                   'PUSH_STR a\nPUSH_STR empty\nSTR_SPLIT\nPUSH_I64 0\nPUSH_STR a\nARR_SET\nPOP\n'):
            asm, mod, output = self.work/'refuse.nasm', self.work/'refuse.nvm', self.work/'old.ll'
            asm.write_text(self.program(op)); self.run_cmd([ROOT/'bin/nanoisa','asm',asm,'-o',mod])
            output.write_text('previous output')
            self.run_cmd([ROOT/'bin/nvm2llvm',mod,'-o',output],success=False)
            self.assertEqual(output.read_text(), 'previous output')

if __name__ == '__main__': unittest.main()
