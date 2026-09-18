"""I compare exact parser values and the declared cross-host NaN normalization."""
import decimal
import json
import math
import random
import struct
import unittest
from pathlib import Path
from tests import test_llvm_managed_strings as managed
ROOT = managed.ROOT


def ordinary_inputs():
    values=[b'',b'-',b'+',b'.',b'-.word',b'-0',b'+0',b'  -42.5tail',b'\t\n\r\v\f +.5',
            b'1.',b'1e',b'1e+',b'0x',b'0x1',b'0x.8',b'0x1p',b'0x1p+',b'-0x0',
            b'1.25\x0099',b'\x001.2',b'INF',b'-infinity',b'infinit',b'NaN',b'-nan',
            b'nan(123)',b'nan(0123)',b'nan(0x123)',b'nan(foo)',b'nan(123foo)',b'nan(09)',
            b'nan(184467440737095516160000)',b'nan(0xffffffffffffffffffff)',b'nan(+1)',b'nan()',
            b'2.4703282292062327e-324',b'2.4703282292062328e-324',
            b'1.7976931348623157e308',b'1.7976931348623159e308',
            b'0x1p-1075',b'0x1.0000000000000000000000000000000000001p-1075',
            b'1e9999999999999999999999',b'-1e-9999999999999999999999',
            b'1'+b'0'*2000+b'e-2000',b'0.'+b'0'*2000+b'1e2001']
    with decimal.localcontext() as context:
        context.prec=3000
        D=decimal.Decimal
        for midpoint in [D(1)/(D(2)**1075),D(2**53-1)/(D(2)**1075),
                         D(2**53+1)/D(2**53),D(2**53+3)/D(2**53),D(2**54-1)*D(2**970)]:
            for delta in [D(0),D(10)**-1400,-(D(10)**-1400)]:
                value=midpoint+delta
                values += [format(value,'f').encode(),format(-value,'f').encode()]
    randoms=random.Random(0x4D2F)
    for i in range(512):
        value=struct.unpack('d',struct.pack('Q',randoms.getrandbits(64)))[0]
        if math.isfinite(value):
            values += [value.hex().encode(),format(value,'.17g').encode()]
            if i<32: values.append(format(value,'.200g').encode())
    return values


def c_bytes(value):
    return '"'+''.join('\\x%02x'%byte for byte in value)+'"'


class Binary64Parse(unittest.TestCase):
    setUp=managed.ManagedStrings.setUp
    run_cmd=managed.ManagedStrings.run_cmd
    program=managed.ManagedStrings.program
    compile=managed.ManagedStrings.compile
    native_harness=managed.ManagedStrings.native_harness
    node=managed.ManagedStrings.node

    def reference(self):
        values=ordinary_inputs()
        header=self.work/'inputs.h'
        header.write_text('static const struct {const unsigned char *text;uint32_t length;} inputs[]={\n'+
                          ''.join('{(const unsigned char*)'+c_bytes(text)+','+str(len(text))+'},\n' for text in values)+'};\n')
        source,exe=self.work/'reference.c',self.work/'reference'
        source.write_text('''#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <fenv.h>
#include "inputs.h"
int main(void){if(sizeof(double)!=8||DBL_MANT_DIG!=53||fegetround()!=FE_TONEAREST)return 1;
for(unsigned i=0;i<sizeof inputs/sizeof inputs[0];i++){double value=strtod((const char*)inputs[i].text,0);uint64_t bits;memcpy(&bits,&value,8);printf("%016llx\\n",(unsigned long long)bits);}return 0;}
''')
        self.run_cmd(self.clang+[source,'-lm','-o',exe])
        expected=[int(line,16) for line in self.run_cmd([exe]).stdout.splitlines()]
        self.assertEqual(len(expected),len(values))
        # I deliberately normalize the observed Darwin overflowing-decimal NaN
        # payload difference to the documented existing Linux saturation policy.
        expected[values.index(b'nan(184467440737095516160000)')]=0x7fffffffffffffff
        (self.work/'expected.h').write_text('static const uint64_t expected[]={'+
                                          ','.join('UINT64_C(0x%016x)'%bits for bits in expected)+'};\n')
        return values,expected

    def core_source(self):
        source=self.work/'core.c'
        source.write_text('''#include "managed_strings.h"
#include "inputs.h"
#include "expected.h"
int run(void){NmsRuntime r;nms_init(&r,0,0);
for(unsigned i=0;i<sizeof inputs/sizeof inputs[0];i++){
 NmsHandle handle;uint64_t bits=0;
 if(nms_create(&r,inputs[i].text,inputs[i].length,&handle)!=NMS_OK)return 10000+i;
 if(nms_retain(&r,handle)!=NMS_OK)return 20000+i;
#ifdef NMS_TESTING
 nms_test_fail_after(&r,0);
#endif
 if(nms_parse_f64(&r,handle,&bits)!=NMS_OK||bits!=expected[i])return 30000+i;
 if(r.live_objects!=1||r.slots[handle&~NMS_DYNAMIC].references!=2)return 40000+i;
 if(nms_release(&r,handle)!=NMS_OK||nms_release(&r,handle)!=NMS_OK||r.live_objects)return 50000+i;
#ifdef NMS_TESTING
 nms_test_fail_after(&r,UINT64_MAX);
#endif
}return nms_dispose(&r);}
#ifndef __wasm32__
#include <stdio.h>
int main(void){int result=run();if(result)fprintf(stderr,"parser reference failure %d\\n",result);return result?1:0;}
#endif
''')
        return source

    def test_core_bits_and_ownership_match_reference_on_native_and_wasm(self):
        values,_=self.reference()
        self.assertGreater(len(values),1100)
        source=self.core_source()
        core=ROOT/'src/nanoisa/managed_strings.c'
        include=['-I'+str(ROOT/'src/nanoisa'),'-I'+str(self.work)]
        exe=self.work/'core'
        self.run_cmd(self.clang+['-DNMS_TESTING','-O2','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',*include,core,source,'-o',exe])
        self.run_cmd([exe])
        wasm=self.work/'core.wasm'
        self.run_cmd(['clang','--target=wasm32-unknown-unknown','-DNMS_TESTING','-O2','-ffreestanding','-fno-builtin','-nostdlib',*include,core,source,'-Wl,--no-entry','-Wl,--export=run','-o',wasm])
        self.node(wasm,'check(e.run()===0);check(e.run()===0);')
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','run',wasm]).stdout,'0\n')

    def test_vm_bits_and_both_generated_c_conversion_paths(self):
        values,expected=self.reference()
        inputs=self.work/'inputs.bin'
        inputs.write_bytes(b''.join(struct.pack('<I',len(value))+value for value in values))
        actual=self.run_cmd([ROOT/'obj/binary64_parser_vm',inputs]).stdout.splitlines()
        self.assertEqual([int(value,16) for value in actual],expected)
        module,_,_=self.compile('.string a "1.25"\n.entry main\n.function main 0 4 0 int 1\nPUSH_STR a\nCAST_FLOAT\nPOP\nPUSH_STR a\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nCAST_FLOAT\nPOP\nPUSH_I64 0\nRET\n.end\n')
        generated=self.work/'program.c'
        self.run_cmd([ROOT/'bin/nvm2c',module,'-o',generated])
        source,exe=self.work/'native.c',self.work/'native'
        source.write_text('''#define main generated_main
#include "program.c"
#undef main
#include "inputs.h"
#include "expected.h"
int main(void){if(generated_main())return 1;
for(unsigned i=0;i<sizeof inputs/sizeof inputs[0];i++){
 double direct=nparse_binary64((const char*)inputs[i].text);
 double boxed=nvalue_cast_float((nmap_value){5,0,(const char*)inputs[i].text});
 uint64_t a,b;memcpy(&a,&direct,8);memcpy(&b,&boxed,8);
 if(a!=expected[i]||b!=expected[i]){fprintf(stderr,"native parser reference %u\\n",i);return 2;}
}return 0;}
''')
        self.run_cmd(self.clang+['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-lm','-o',exe])
        self.run_cmd([exe])

    def test_emitted_float_parsing_aliases_and_cleanup(self):
        values,expected=self.reference()
        strings,body='',''
        for i,(text,bits) in enumerate(zip(values[:75],expected[:75])):
            value=struct.unpack('d',struct.pack('Q',bits))[0]
            strings+=f'.string s{i} '+c_bytes(text)+'\n'
            body+=f'PUSH_STR s{i}\nPUSH_STR empty\nSTR_CONCAT\nDUP\nSTORE_GLOBAL 0\nCALL parse\n'
            if math.isnan(value):
                body+='DUP\nNE\nASSERT\n'
            else:
                body+=f'PUSH_F64 {value.hex()}\nEQ\nASSERT\n'
        suffix='.function parse 1 1 0 float 1\n.parameters parse string\nLOAD_LOCAL 0\nCAST_FLOAT\nRET\n.end\n'
        _,ir,wasm=self.compile(strings+self.program(body,suffix))
        self.native_harness(ir,'for(int i=0;i<4;i++)if(nano_try_entry()||nms_module_live_objects()!=1)return 1;return nano_dispose();')
        self.node(wasm,'for(let i=0;i<4;i++){check(e.nano_try_entry()===0n);check(e.nms_module_live_objects()===1n);}check(e.nano_dispose()===0);')


if __name__=='__main__':
    unittest.main()
