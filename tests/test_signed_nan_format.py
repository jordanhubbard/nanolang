"""I preserve exact nonfinite signs across VM/native/managed conversion."""
import json
from pathlib import Path
import unittest
from tests import test_llvm_managed_strings as managed
ROOT = managed.ROOT
CASES = [(0x7ff8000000000001,'nan'),(0xfff8000000000001,'-nan'),
         (0x7ff0000000000001,'nan'),(0xfff0000000000001,'-nan'),
         (0x7ff0000000000000,'inf'),(0xfff0000000000000,'-inf'),
         (0,'0'),(1<<63,'-0'),(0x7fefffffffffffff,'1.79769e+308'),
         (0xffefffffffffffff,'-1.79769e+308'),(1,'4.94066e-324')]

class SignedNanFormat(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def test_exact_bit_vm_native_and_managed_conversion(self):
        strings='';body=''
        for i,(bits,expected) in enumerate(CASES):
            strings+=f'.string expected{i} {json.dumps(expected)}\n'
            body+=f'PUSH_F64 bits:{bits:016x}\nDUP\nF64_TO_BITS\nPUSH_I64 {bits if bits<1<<63 else bits-(1<<64)}\nI64_EQ\nASSERT\nCAST_STRING\nPUSH_STR expected{i}\nEQ\nASSERT\n'
        module,ir,wasm=self.compile(strings+self.program(body))
        self.native_harness(ir,'if(nano_try_entry()||nms_module_live_objects())return 1;return nano_dispose();')
        self.node(wasm,'check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);check(e.nano_dispose()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_entry',wasm]).stdout,'0\n')
        c=self.work/'native.c';exe=self.work/'native'
        self.run_cmd([ROOT/'bin/nvm2c',module,'-o',c])
        self.run_cmd(self.clang+['-O2','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all',c,'-lm','-o',exe])
        self.run_cmd([exe])

    def test_vm_and_generated_print(self):
        assembly=self.work/'print.nasm';module=self.work/'print.nvm'
        assembly.write_text(self.program(''.join(f'PUSH_F64 bits:{bits:016x}\nPRINTLN\n' for bits,_ in CASES)))
        self.run_cmd([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        expected='\n'.join(text+'.0' if bits in (0,1<<63) else text for bits,text in CASES)+'\n'
        self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,expected)
        c=self.work/'print.c';exe=self.work/'print'
        self.run_cmd([ROOT/'bin/nvm2c',module,'-o',c])
        self.run_cmd(self.clang+['-O2','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all',c,'-lm','-o',exe])
        self.assertEqual(self.run_cmd([exe]).stdout,expected)

    def test_direct_vm_display_and_shared_helper(self):
        source=self.work/'display.c';exe=self.work/'display'
        rows=''.join('{UINT64_C(0x%016x),%s},\n'%(bits,json.dumps(text)) for bits,text in CASES)
        source.write_text('''#include "src/binary64_format.h"
#include "src/nanovm/value.h"
#include "src/nanovm/heap.h"
#include <stdlib.h>
static const struct {uint64_t bits;const char *text;} cases[]={'''+rows+'''};
int main(void){
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++){
  double x;memcpy(&x,&cases[i].bits,8);char buf[64];
  int n=nano_rt_f64_format(buf,sizeof buf,x);
  if(n!=(int)strlen(cases[i].text)||strcmp(buf,cases[i].text))return 1;
  char small[2];if(nano_rt_f64_format(small,sizeof small,x)!=n||small[1])return 2;
  if(nano_rt_f64_format(NULL,0,x)!=n)return 3;
  NanoValue value=val_float(x);char *text=val_to_cstring(value);
  if(!text||strcmp(text,cases[i].text))return 4;free(text);
  VmHeap heap;vm_heap_init(&heap);VmString *str=vm_string_from_float(&heap,x);
  if(!str||strcmp(vmstring_cstr(str),cases[i].text))return 5;
  vm_release(&heap,val_string(str));vm_heap_destroy(&heap);
  FILE *out=tmpfile();if(!out)return 6;val_print(value,out);
  if(fflush(out)||fseek(out,0,SEEK_SET))return 7;
  size_t got=fread(buf,1,sizeof buf-1,out);buf[got]=0;fclose(out);
  const char *printed=i==6?"0.0":i==7?"-0.0":cases[i].text;
  if(strcmp(buf,printed))return 8;
  uint64_t after;memcpy(&after,&x,8);if(after!=cases[i].bits)return 9;
 }
 return 0;
}
''')
        sources=['src/nanovm/value.c','src/nanovm/heap.c','src/nanovm/heap_cycles.c','src/nanoisa/isa.c']
        for standard in ('c99','c11'):
            self.run_cmd(self.clang+['-D_POSIX_C_SOURCE=200809L','-std='+standard,'-O2','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-I'+str(ROOT),source,*[ROOT/p for p in sources],'-lm','-o',exe])
            self.run_cmd([exe])

if __name__=='__main__':unittest.main()
