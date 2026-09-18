"""I preserve trim bytes, fresh ownership, and cleanup on actual targets."""
import unittest
from pathlib import Path
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes
ROOT=Path(__file__).resolve().parents[1]
CASES=[b'',b' ',b' \t\n\r',b'abc',b' abc ',b'\t\na b\r',b' \x00 ',b'\x00 a\x00',
       b' \xff\x80 ',b'\x0babc\x0c',b' '+b'a'*1024+b'\n',b'  a\x00b\t']
class ManagedTrim(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_fresh_core_results_aliases_and_failure(self):
        rows=''.join('{'+c_bytes(a)+','+str(len(a))+','+c_bytes(a.strip(b' \t\n\r'))+','+str(len(a.strip(b' \t\n\r')))+'},\n' for a in CASES)
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
static const struct{const char *input;uint32_t length;const char *expected;uint32_t size;} cases[]={
'''+rows+'''};
int run(void){NmsRuntime r;nms_init(&r,0,0);
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
  NmsHandle a,out=99;NmsView view;
  if(nms_create(&r,(const unsigned char*)cases[i].input,cases[i].length,&a)||nms_retain(&r,a))return 1;
  if(nms_trim_owned(&r,a,&out)||out==a||r.live_objects!=2)return 2;
  if(nms_view(&r,out,&view)||view.length!=cases[i].size)return 3;
  for(uint32_t k=0;k<view.length;k++)if(view.data[k]!=(unsigned char)cases[i].expected[k])return 4;
  if(nms_release(&r,out)||r.slots[a&~NMS_DYNAMIC].references!=1)return 5;
  if(nms_retain(&r,a))return 6;
  out=99;nms_test_fail_after(&r,0);
  if(nms_trim_owned(&r,a,&out)!=NMS_MEMORY||out!=99||r.live_objects!=1||r.slots[a&~NMS_DYNAMIC].references!=1)return 7;
  if(nms_retain(&r,a)||nms_trim_owned(&r,a,0)!=NMS_STATE||r.slots[a&~NMS_DYNAMIC].references!=1)return 8;
  if(nms_release(&r,a)||r.live_objects||r.live_bytes)return 9;
  nms_test_fail_after(&r,UINT64_MAX);
 }
 return nms_dispose(&r);
}
#ifndef __wasm32__
int main(void){return run();}
#endif
''')
        core=ROOT/'src/nanoisa/managed_strings.c';inc='-I'+str(ROOT/'src/nanoisa')
        native,wasm=self.work/'core',self.work/'core.wasm'
        self.run_cmd(self.clang+['-DNMS_TESTING','-O2','-fsanitize=address,undefined','-fno-sanitize-recover=all',inc,core,source,'-o',native]);self.run_cmd([native])
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-DNMS_TESTING','-O2','-ffreestanding','-fno-builtin','-nostdlib',inc,core,source,'-Wl,--no-entry','-Wl,--export=run','-o',wasm])
        self.node(wasm,'check(e.run()===0);check(e.run()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','run',wasm]).stdout,'0\n')

    def test_exact_bytes_literals_dynamic_calls_globals_reentry(self):
        strings='.string empty ""\n';body=''
        for i,a in enumerate(CASES):
            strings+=f'.string a{i} '+c_bytes(a)+f'\n.string e{i} '+c_bytes(a.strip(b' \t\n\r'))+'\n'
            body+=f'PUSH_STR a{i}\nSTR_TRIM\nPUSH_STR e{i}\nSTR_EQ\nASSERT\n'
            body+=f'PUSH_STR a{i}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL trim\nPUSH_STR e{i}\nEQ\nASSERT\n'
            body+=f'LOAD_GLOBAL 0\nPUSH_STR a{i}\nEQ\nASSERT\n'
        text=strings+'.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n.function trim 1 1 0 string 1\n.parameters trim string\nLOAD_LOCAL 0\nSTR_TRIM\nRET\n.end\n'
        _,ir,wasm=self.compile(text)
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_emitted_allocation_failure_recovery_and_wrong_tag(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL trim\nPOP\n'
        suffix='.function trim 1 1 0 string 1\n.parameters trim string\nLOAD_LOCAL 0\nSTR_TRIM\nRET\n.end\n'
        _,ir,wasm=self.compile(self.program(body,suffix))
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        self.native_harness(ir,'budget=2;if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        bad='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nPUSH_I64 7\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nSTR_TRIM\nPOP\n'
        _,ir,wasm=self.compile(self.program(bad),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
if __name__=='__main__':unittest.main()
