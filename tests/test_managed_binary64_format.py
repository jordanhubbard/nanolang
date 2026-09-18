"""I compare portable binary64 bytes to ordinary native C-locale references."""
import json
from pathlib import Path
import unittest
from tests import test_llvm_managed_strings as managed
ROOT = managed.ROOT

REFERENCE = r'''
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <fenv.h>
static void emit(uint64_t bits) {
    double value; memcpy(&value, &bits, 8);
    printf("%016llx %g %a\n", (unsigned long long)bits, value, value);
}
int main(void) {
    if (sizeof(double)!=8 || DBL_MANT_DIG!=53 || fegetround()!=FE_TONEAREST) return 1;
    double cases[]={0.0,-0.0,1.0,-1.0,123456.5,123457.5,999998.5,999999.5,
      -123456.5,-123457.5,-999999.5,0.00009999995,0.000009999995,0.0001,0.00001,1e6,1e5,
      DBL_MIN,DBL_MAX,0x0.0000000000001p-1022,-0x0.0000000000001p-1022,
      1.234565,1.234575,9.999995,9.999994};
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
      uint64_t bits; memcpy(&bits,&cases[i],8); emit(bits);
    }
    uint64_t special[]={UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
      UINT64_C(0x7ff8000000000001),UINT64_C(0xfff8000000000001)};
    for(unsigned i=0;i<4;i++) emit(special[i]);
    uint64_t bits=UINT64_C(0x3456789abcdef012);
    for(unsigned i=0;i<2048;i++) {bits^=bits<<13;bits^=bits>>7;bits^=bits<<17;emit(bits);}
}
'''


class Binary64Format(unittest.TestCase):
    setUp = managed.ManagedStrings.setUp
    run_cmd = managed.ManagedStrings.run_cmd
    program = managed.ManagedStrings.program
    compile = managed.ManagedStrings.compile
    native_harness = managed.ManagedStrings.native_harness
    node = managed.ManagedStrings.node

    def reference(self):
        source,exe=self.work/'reference.c',self.work/'reference'
        source.write_text(REFERENCE)
        self.run_cmd(self.clang+[source,'-lm','-o',exe])
        return [line.split() for line in self.run_cmd([exe]).stdout.splitlines()]

    def test_portable_core_matches_2077_reference_values(self):
        values=self.reference()
        self.assertEqual(len(values),2077)
        header=self.work/'expected.h'
        header.write_text('static const struct {uint64_t bits;const char *text;} cases[]={\n'+
                          ''.join('{UINT64_C(0x'+bits+'),'+json.dumps(text)+'},\n' for bits,text,_ in values)+'};\n')
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
#include "expected.h"
int run(void) {
 NmsRuntime r; nms_init(&r,0,0);
 for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++) {
  NmsHandle out=0; NmsView v;
  if(nms_format_scalar(&r,cases[i].bits,3,&out)!=NMS_OK || nms_view(&r,out,&v)!=NMS_OK)return 10000+i;
  unsigned n=0;while(cases[i].text[n])n++;
  if(v.length!=n)return 20000+i;
  for(unsigned j=0;j<n;j++)if(v.data[j]!=(unsigned char)cases[i].text[j])return 30000+i;
  if(nms_release(&r,out)!=NMS_OK || r.live_objects)return 40000+i;
 }
 return nms_dispose(&r);
}
#ifndef __wasm32__
#include <stdio.h>
int main(void){int result=run();if(result)fprintf(stderr,"reference failure %d\\n",result);return result?1:0;}
#endif
''')
        core=ROOT/'src/nanoisa/managed_strings.c'
        include=['-I'+str(ROOT/'src/nanoisa'),'-I'+str(self.work)]
        exe=self.work/'core'
        self.run_cmd(self.clang+['-O2','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',*include,core,source,'-o',exe])
        self.run_cmd([exe])
        wasm=self.work/'core.wasm'
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-O2','-ffreestanding','-fno-builtin','-nostdlib',*include,core,source,
                      '-Wl,--no-entry','-Wl,--export=run','-o',wasm])
        self.node(wasm,'check(e.run()===0);check(e.run()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','run',wasm]).stdout,'0\n')

    def test_emitted_formatting_matches_vm_and_reference(self):
        values=self.reference()[:29]
        strings,body='',''
        for i,(_,expected,literal) in enumerate(values):
            strings+=f'.string e{i} "{expected}"\n'
            body+=f'PUSH_F64 {literal}\nCALL format\nPUSH_STR e{i}\nEQ\nASSERT\n'
        body+='PUSH_F64 123456.5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCAST_STRING\nSTORE_GLOBAL 1\n'
        suffix=('.function format 1 1 0 string 1\n.parameters format float\n'
                'LOAD_LOCAL 0\nCAST_STRING\nRET\n.end\n')
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<10;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<10;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);')

    def test_float_format_allocation_failure_recovers(self):
        _,ir,wasm=self.compile(self.program('PUSH_F64 0x1.fffffffffffffp+1023\nCAST_STRING\nPOP\n'))
        extra='static long budget=-1;extern void *__real_malloc(size_t);void *__wrap_malloc(size_t n){if(!budget)return 0;if(budget>0)--budget;return __real_malloc(n);}'
        for fail in (0,1):
            self.native_harness(ir,f'budget={fail};if(nano_try_entry()!=((uint64_t)3<<32)||nms_module_live_objects())return 1;budget=-1;if(nano_try_entry()||nms_module_live_objects())return 2;return nano_dispose();',extra,['-Wl,--wrap=malloc'])
        self.node(wasm,'check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===0n);check(e.nano_dispose()===0);')


if __name__=='__main__':
    unittest.main()
