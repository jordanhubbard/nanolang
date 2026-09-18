"""I preserve byte predicates and borrowed-owner cleanup on actual targets."""
import unittest
from pathlib import Path
from tests import test_llvm_managed_strings as managed
from tests.test_managed_binary64_parse import c_bytes

ROOT = Path(__file__).resolve().parents[1]
CASES = [(b'',b''),(b'',b'a'),(b'abc',b''),(b'abc',b'abc'),(b'abc',b'ab'),
         (b'abc',b'bc'),(b'abc',b'b'),(b'abc',b'abcd'),(b'abc',b'ac'),
         (b'a\x00bc',b'\x00b'),(b'a\x00bc',b'bc'),(b'a\x00bc',b'a\x00'),
         (b'\x00',b'\x00'),(b'abc',b'\x00'),(b'\xff\x80a',b'\xff\x80'),
         (b'\xff\x80a',b'\x80a'),(b'abababac',b'ababac'),(b'a'*1024+b'b',b'aaaaab')]
OPS = ('STR_CONTAINS','STR_STARTS_WITH','STR_ENDS_WITH')
def answers(left,right): return (right in left,left.startswith(right),left.endswith(right))

class ManagedPredicates(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_borrowed_runtime_predicates_allocate_nothing(self):
        source=self.work/'core.c'
        rows=''.join('{'+c_bytes(a)+','+str(len(a))+','+c_bytes(b)+','+str(len(b))+',{' +
                     ','.join(str(int(x)) for x in answers(a,b))+'}},\n' for a,b in CASES)
        source.write_text('''#include "managed_strings.h"
static const struct {const char *a;uint32_t al;const char *b;uint32_t bl;uint32_t answer[3];} cases[]={
'''+rows+'''};
int run(void){NmsRuntime r;nms_init(&r,0,0);
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
  NmsHandle a,b;uint32_t answer=99;
  if(nms_create(&r,(const unsigned char*)cases[i].a,cases[i].al,&a) ||
     nms_create(&r,(const unsigned char*)cases[i].b,cases[i].bl,&b))return 1;
  if(nms_retain(&r,a)||nms_retain(&r,b))return 2;
  nms_test_fail_after(&r,0);
  for(uint32_t mode=0;mode<3;mode++){
   if(nms_predicate(&r,a,b,mode,&answer)||answer!=cases[i].answer[mode])return 3;
   if(nms_predicate(&r,a,a,mode,&answer)||answer!=1)return 4;
  }
  answer=99;if(nms_predicate(&r,a,b,3,&answer)!=NMS_STATE||answer!=99)return 5;
  if(nms_predicate(&r,a,b,NMS_CONTAINS,0)!=NMS_STATE||answer!=99)return 5;
  if(r.live_objects!=2||r.slots[a&~NMS_DYNAMIC].references!=2||r.slots[b&~NMS_DYNAMIC].references!=2)return 6;
  if(nms_release(&r,a)||nms_release(&r,a)||nms_release(&r,b)||nms_release(&r,b)||r.live_objects)return 7;
  nms_test_fail_after(&r,UINT64_MAX);
 }
 return nms_dispose(&r);
}
#ifndef __wasm32__
int main(void){return run();}
#endif
''')
        core=ROOT/'src/nanoisa/managed_strings.c'
        include='-I'+str(ROOT/'src/nanoisa')
        native,wasm=self.work/'native',self.work/'core.wasm'
        self.run_cmd(self.clang+['-DNMS_TESTING','-O2','-fsanitize=address,undefined','-fno-sanitize-recover=all',include,core,source,'-o',native])
        self.run_cmd([native])
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-DNMS_TESTING','-O2','-ffreestanding','-fno-builtin','-nostdlib',include,core,source,'-Wl,--no-entry','-Wl,--export=run','-o',wasm])
        self.node(wasm,'check(e.run()===0);check(e.run()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','run',wasm]).stdout,'0\n')

    def test_actual_byte_predicates_calls_globals_and_aliases(self):
        strings='.string empty ""\n'
        body=''
        for i,(left,right) in enumerate(CASES):
            strings+=f'.string a{i} '+c_bytes(left)+f'\n.string b{i} '+c_bytes(right)+'\n'
            for mode,expected in enumerate(answers(left,right)):
                body+=f'PUSH_STR a{i}\nPUSH_STR b{i}\nCALL predicate{mode}\n'
                body+=('' if expected else 'NOT\n')+'ASSERT\n'
                body+=f'PUSH_STR a{i}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nPUSH_STR b{i}\nPUSH_STR empty\nSTR_CONCAT\nCALL predicate{mode}\n'
                body+=('' if expected else 'NOT\n')+'ASSERT\n'
                body+=f'LOAD_GLOBAL 0\nDUP\nCALL predicate{mode}\nASSERT\n'
        text=strings+'.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'
        for mode,op in enumerate(OPS):
            text+=f'.function predicate{mode} 2 2 0 bool 1\n.parameters predicate{mode} string string\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n{op}\nRET\n.end\n'
        _,ir,wasm=self.compile(text)
        self.native_harness(ir,'for(int i=0;i<4;i++){if(nano_try_entry()!=0||nms_module_live_objects()!=1)return 1;}if(nano_dispose()||nms_module_live_objects())return 2;return 0;')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')

    def test_dynamic_wrong_tags_clean_operands_and_keep_globals(self):
        for op in OPS:
            for reverse in (False,True):
                with self.subTest(op=op,reverse=reverse):
                    body='PUSH_STR a\nPUSH_STR empty\nSTR_CONCAT\nSTORE_GLOBAL 0\nPUSH_I64 7\nSTORE_GLOBAL 1\n'
                    body+=('LOAD_GLOBAL 1\nLOAD_GLOBAL 0\n' if reverse else 'LOAD_GLOBAL 0\nLOAD_GLOBAL 1\n')
                    body+=op+'\nPOP\n'
                    _,ir,wasm=self.compile(self.program(body),vm_ok=False)
                    self.native_harness(ir,'for(int i=0;i<2;i++){if((nano_try_entry()>>32)!=1||nms_module_live_objects()!=1)return 1;}if(nano_dispose()||nms_module_live_objects())return 2;return 0;')
                    self.node(wasm,'for(let i=0;i<2;i++){check((e.nano_try_entry()>>32n)===1n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);check(e.nms_module_live_objects()===0n);')

if __name__=='__main__': unittest.main()
