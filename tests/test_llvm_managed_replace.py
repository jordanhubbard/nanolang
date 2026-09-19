"""I preserve exact replacement bytes and consuming alias cleanup on real targets."""
import unittest
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes
ROOT=managed.ROOT
CASES=[(b'abcabc',b'bc',b''),(b'abab',b'ab',b'XYZ'),(b'a\x00ba\x00b',b'\x00b',b'Z\x00'),
       (b'aaaaa',b'aa',b'b'),(b'abc',b'x',b'y'),(b'abc',b'',b'XYZ'),(b'',b'a',b'b'),
       (b'aaa',b'a',b''),(b'\xffa\x80a',b'a',b'\x00\xff'),(b'a'*600,b'a',b'BC')]
def replaced(a,b,c):return a.replace(b,c) if b else a
class ManagedReplace(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_core_allocation_stages_growth_and_equal_handle_owners(self):
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
int run(void){NmsRuntime r;nms_init(&r,0,0);NmsHandle held[8],out=99;NmsView v;
 const char *text[8]={"abab","ab","XYZ","f","g","h","i","j"};
 const uint32_t lengths[8]={4,2,3,1,1,1,1,1};
 for(unsigned i=0;i<8;i++)if(nms_create(&r,(const unsigned char*)text[i],lengths[i],&held[i]))return 1;
 uint64_t allocations=nms_test_live_allocations();
 for(unsigned fail=0;fail<3;fail++){
  for(unsigned i=0;i<3;i++)if(nms_retain(&r,held[i]))return 2;
  nms_test_fail_after(&r,fail);out=99;
  if(nms_replace_owned(&r,held[0],held[1],held[2],&out)!=NMS_MEMORY||out!=99)return 3;
  if(r.capacity!=8||r.live_objects!=8||nms_test_live_allocations()!=allocations)return 4;
  for(unsigned i=0;i<3;i++)if(r.slots[held[i]&~NMS_DYNAMIC].references!=1)return 5;
  nms_test_fail_after(&r,UINT64_MAX);
 }
 for(unsigned i=0;i<3;i++)if(nms_retain(&r,held[i]))return 6;
 if(nms_replace_owned(&r,held[0],held[1],held[2],&out)||r.capacity!=16)return 7;
 if(nms_view(&r,out,&v)||v.length!=6)return 8;
 for(unsigned i=0;i<6;i++)if(v.data[i]!=(unsigned char)"XYZXYZ"[i])return 9;
 if(nms_release(&r,out))return 10;
 for(unsigned i=0;i<3;i++)if(nms_retain(&r,held[0]))return 11;
 if(nms_replace_owned(&r,held[0],held[0],held[0],&out)||out==held[0])return 12;
 if(nms_view(&r,out,&v)||v.length!=4||v.data[0]!='a'||v.data[3]!='b')return 13;
 if(nms_release(&r,out)||r.slots[held[0]&~NMS_DYNAMIC].references!=1)return 14;
 for(unsigned i=0;i<3;i++)if(nms_retain(&r,held[i]))return 15;
 if(nms_replace_owned(&r,held[0],held[1],held[2],0)!=NMS_STATE)return 16;
 for(unsigned i=0;i<8;i++){
  if(nms_view(&r,held[i],&v)||v.length!=lengths[i]||r.slots[held[i]&~NMS_DYNAMIC].references!=1)return 17;
  for(unsigned k=0;k<v.length;k++)if(v.data[k]!=(unsigned char)text[i][k])return 18;
  if(nms_release(&r,held[i]))return 19;
 }
 if(r.live_objects||r.live_bytes||nms_dispose(&r)||nms_test_live_allocations())return 20;
 return 0;
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

    def test_bytes_aliases_calls_globals_and_reentry(self):
        strings='';body=''
        for i,(a,b,c) in enumerate(CASES):
            for label,value in [('a',a),('b',b),('c',c),('e',replaced(a,b,c))]:strings+=f'.string {label}{i} '+c_bytes(value)+'\n'
            body+=f'PUSH_STR a{i}\nPUSH_STR b{i}\nPUSH_STR c{i}\nSTR_REPLACE\nPUSH_STR e{i}\nEQ\nASSERT\n'
            for slot,label in enumerate(('a','b','c')):body+=f'PUSH_STR {label}{i}\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL {slot}\n'
            body+=f'LOAD_GLOBAL 0\nLOAD_GLOBAL 1\nLOAD_GLOBAL 2\nCALL replace\nPUSH_STR e{i}\nEQ\nASSERT\n'
            for slot,label in enumerate(('a','b','c')):body+=f'LOAD_GLOBAL {slot}\nPUSH_STR {label}{i}\nEQ\nASSERT\n'
            body+=f'LOAD_GLOBAL 0\nDUP\nDUP\nSTR_REPLACE\nPUSH_STR a{i}\nEQ\nASSERT\n'
            body+=f'LOAD_GLOBAL 0\nDUP\nLOAD_GLOBAL 2\nSTR_REPLACE\nPUSH_STR {"c" if a else "a"}{i}\nEQ\nASSERT\n'
            body+=f'LOAD_GLOBAL 0\nLOAD_GLOBAL 1\nDUP\nSTR_REPLACE\nPUSH_STR a{i}\nEQ\nASSERT\n'
            # Source/replacement alias with a separate nonempty needle.
            if b and len(a)<=32:
                strings+=f'.string alias{i} '+c_bytes(replaced(a,b,a))+'\n'
                body+=f'LOAD_GLOBAL 0\nLOAD_GLOBAL 1\nLOAD_GLOBAL 0\nSTR_REPLACE\nPUSH_STR alias{i}\nEQ\nASSERT\n'
        suffix='.function replace 3 3 0 string 1\n.parameters replace string string string\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nSTR_REPLACE\nRET\n.end\n'
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=3)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===3n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_emitted_failure_recovery_and_all_wrong_tag_positions(self):
        body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nPUSH_STR a\nPUSH_STR double\nCALL replace\nPOP\n'
        suffix='.function replace 3 3 0 string 1\n.parameters replace string string string\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nSTR_REPLACE\nRET\n.end\n'
        _,ir,_=self.compile(self.program(body,suffix))
        extra='static long budget=-1;void *nano_test_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return malloc(n);}'
        for fail in (2,3):
            self.native_harness(ir,f'budget={fail};if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects()!=1)return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects()!=1)return 2;return nano_dispose();',extra,allocation_control=True)
        for bad_position in range(3):
            body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nPUSH_I64 7\nSTORE_GLOBAL 1\n'
            body+=''.join('LOAD_GLOBAL 1\n' if i==bad_position else 'LOAD_GLOBAL 0\n' for i in range(3))+'STR_REPLACE\nPOP\n'
            _,ir,wasm=self.compile(self.program(body),vm_ok=False)
            self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()!=((uint64_t)1<<32)||nms_module_live_objects()!=1)return 1;return nano_dispose();')
            self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===(1n<<32n));check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
if __name__=='__main__':unittest.main()
