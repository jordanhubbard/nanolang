"""I match stored-byte character access and two-operand cleanup on actual targets."""
import unittest
from pathlib import Path
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes
ROOT=Path(__file__).resolve().parents[1]
CASES=[b'',b'a',b'a\x00\xff\x80z',b'\x00',b'\xff',b'a'*1024+b'\x80']
INDICES=[-(1<<63),-1,0,1,2,4,1024,1025,(1<<63)-1]
class ManagedCharacter(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_borrowed_core_bytes_no_allocation_and_reference_stability(self):
        rows=''.join('{'+c_bytes(a)+','+str(len(a))+',UINT64_C('+str(i&((1<<64)-1))+'),'+str(a[i] if 0<=i<len(a) else -1)+'},\n' for a in CASES for i in INDICES)
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
static const struct{const char *bytes;uint32_t length;uint64_t index;int64_t expected;} cases[]={
'''+rows+'''};
int run(void){NmsRuntime r;nms_init(&r,0,0);
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
  NmsHandle a;int64_t answer=999;
  if(nms_create(&r,(const unsigned char*)cases[i].bytes,cases[i].length,&a)||nms_retain(&r,a))return 1;
  nms_test_fail_after(&r,0);
  if(nms_char_at(&r,a,cases[i].index,1,&answer)||answer!=cases[i].expected)return 2;
  if(nms_char_at(&r,a,UINT64_MAX,0,&answer)||answer!=(cases[i].length?(unsigned char)cases[i].bytes[0]:-1))return 3;
  answer=999;if(nms_char_at(&r,a,0,2,&answer)!=NMS_STATE||answer!=999)return 4;
  if(nms_char_at(&r,a,0,1,0)!=NMS_STATE)return 5;
  if(r.live_objects!=1||r.live_bytes!=cases[i].length||r.slots[a&~NMS_DYNAMIC].references!=2)return 6;
  if(nms_release(&r,a)||nms_release(&r,a)||r.live_objects||r.live_bytes)return 7;
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

    def test_exact_bytes_indices_calls_globals_and_nonnumeric_fallback(self):
        strings='.string empty ""\n';body=''
        for n,a in enumerate(CASES):
            strings+=f'.string a{n} '+c_bytes(a)+'\n'
            for i in INDICES:
                expected=a[i] if 0<=i<len(a) else -1
                body+=f'PUSH_STR a{n}\nPUSH_I64 {i}\nSTR_CHAR_AT\nPUSH_I64 {expected}\nEQ\nASSERT\n'
                body+=f'PUSH_STR a{n}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nPUSH_I64 {i}\nCALL byte_at\nPUSH_I64 {expected}\nEQ\nASSERT\n'
            # Both arguments can be aliases of one dynamic owner. The string
            # index is an existing VM fallback, not a source typing extension.
            expected=a[0] if a else -1
            body+=f'LOAD_GLOBAL 0\nDUP\nSTR_CHAR_AT\nPUSH_I64 {expected}\nEQ\nASSERT\n'
            for instruction in ('PUSH_BOOL 1','PUSH_U8 255','PUSH_F64 2.5','PUSH_VOID','ENUM_VAL 0 2'):
                body+=f'LOAD_GLOBAL 0\n{instruction}\nSTR_CHAR_AT\nPUSH_I64 {expected}\nEQ\nASSERT\n'
            body+=f'LOAD_GLOBAL 0\nPUSH_STR a{n}\nEQ\nASSERT\n'
        text=strings+'.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n.function byte_at 2 2 0 int 1\n.parameters byte_at string int\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nSTR_CHAR_AT\nRET\n.end\n'
        _,ir,wasm=self.compile(text)
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_wrong_source_releases_string_index_and_preserves_globals(self):
        bad='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nPUSH_I64 7\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nLOAD_GLOBAL 0\nSTR_CHAR_AT\nPOP\n'
        _,ir,wasm=self.compile(self.program(bad),vm_ok=False)
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
if __name__=='__main__':unittest.main()
