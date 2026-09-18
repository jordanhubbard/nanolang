"""I match ASCII case bytes while preserving managed ownership boundaries."""
import unittest
from pathlib import Path
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes
ROOT=Path(__file__).resolve().parents[1]
CASES=[b'',b'123 \x00\xff',b'AbZy',bytes(range(256)),b'aBzY'*300]
class ManagedCase(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_core_fresh_aliases_failures_and_descriptor_growth(self):
        rows=''.join('{'+c_bytes(a)+','+str(len(a))+','+c_bytes(a.upper() if upper else a.lower())+','+str(upper)+'},\n' for a in CASES for upper in (0,1))
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
static const struct{const char *input;uint32_t size;const char *expected;uint32_t upper;} cases[]={
'''+rows+'''};
int run(void){NmsRuntime r;nms_init(&r,0,0);
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
  NmsHandle a,out=99;NmsView v;
  if(nms_create(&r,(const unsigned char*)cases[i].input,cases[i].size,&a)||nms_retain(&r,a))return 1;
  if(nms_case_owned(&r,a,cases[i].upper,&out)||out==a||r.live_objects!=2)return 2;
  if(nms_view(&r,out,&v)||v.length!=cases[i].size)return 3;
  for(uint32_t k=0;k<v.length;k++)if(v.data[k]!=(unsigned char)cases[i].expected[k])return 4;
  if(nms_release(&r,out)||nms_view(&r,a,&v))return 5;
  for(uint32_t k=0;k<v.length;k++)if(v.data[k]!=(unsigned char)cases[i].input[k])return 6;
  if(nms_retain(&r,a))return 7;
  out=99;nms_test_fail_after(&r,0);
  if(nms_case_owned(&r,a,cases[i].upper,&out)!=NMS_MEMORY||out!=99)return 8;
  if(nms_retain(&r,a)||nms_case_owned(&r,a,2,&out)!=NMS_STATE||out!=99)return 9;
  if(nms_retain(&r,a)||nms_case_owned(&r,a,0,0)!=NMS_STATE)return 10;
  if(r.live_objects!=1||r.slots[a&~NMS_DYNAMIC].references!=1)return 11;
  if(nms_release(&r,a)||r.live_objects||r.live_bytes)return 12;
  nms_test_fail_after(&r,UINT64_MAX);
 }
 if(nms_dispose(&r))return 13;
 nms_init(&r,0,0);
 NmsHandle held[8],out=99;NmsView v;
 for(unsigned i=0;i<8;i++)if(nms_create(&r,(const unsigned char*)"Az",2,&held[i]))return 14;
 if(r.capacity!=8||r.free_head||nms_retain(&r,held[0]))return 15;
 nms_test_fail_after(&r,1); /* Result bytes succeed, replacement table fails. */
 if(nms_case_owned(&r,held[0],0,&out)!=NMS_MEMORY||out!=99||r.capacity!=8||r.live_objects!=8)return 16;
 nms_test_fail_after(&r,UINT64_MAX);
 if(nms_retain(&r,held[0])||nms_case_owned(&r,held[0],0,&out)||r.capacity!=16)return 17;
 if(nms_view(&r,out,&v)||v.length!=2||v.data[0]!='a'||v.data[1]!='z')return 18;
 if(nms_release(&r,out))return 19;
 for(unsigned i=0;i<8;i++){
  if(nms_view(&r,held[i],&v)||v.data[0]!='A'||v.data[1]!='z'||r.slots[held[i]&~NMS_DYNAMIC].references!=1)return 20;
  if(nms_release(&r,held[i]))return 21;
 }
 if(r.live_objects||r.live_bytes)return 22;
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

    def test_all_bytes_literals_dynamic_calls_globals_and_reentry(self):
        strings='.string empty ""\n';body=''
        for i,a in enumerate(CASES):
            strings+=f'.string a{i} '+c_bytes(a)+'\n'
            for upper in (0,1):
                op='STR_TO_UPPER' if upper else 'STR_TO_LOWER'
                strings+=f'.string e{i}_{upper} '+c_bytes(a.upper() if upper else a.lower())+'\n'
                body+=f'PUSH_STR a{i}\n{op}\nPUSH_STR e{i}_{upper}\nEQ\nASSERT\n'
                body+=f'PUSH_STR a{i}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL case{upper}\nPUSH_STR e{i}_{upper}\nEQ\nASSERT\n'
                body+=f'LOAD_GLOBAL 0\nPUSH_STR a{i}\nEQ\nASSERT\n'
        text=strings+'.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'
        for upper in (0,1):
            op='STR_TO_UPPER' if upper else 'STR_TO_LOWER'
            text+=f'.function case{upper} 1 1 0 string 1\n.parameters case{upper} string\nLOAD_LOCAL 0\n{op}\nRET\n.end\n'
        _,ir,wasm=self.compile(text)
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_emitted_allocation_failure_recovery_and_wrong_tag(self):
        for op in ('STR_TO_LOWER','STR_TO_UPPER'):
            body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL change\nPOP\n'
            suffix=f'.function change 1 1 0 string 1\n.parameters change string\nLOAD_LOCAL 0\n{op}\nRET\n.end\n'
            _,ir,wasm=self.compile(self.program(body,suffix))
            extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
            self.native_harness(ir,'budget=2;if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1||nms_module_live_bytes()!=3)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
            bad=f'PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nPUSH_I64 7\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\n{op}\nPOP\n'
            _,ir,wasm=self.compile(self.program(bad),vm_ok=False)
            self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
            self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
if __name__=='__main__':unittest.main()
