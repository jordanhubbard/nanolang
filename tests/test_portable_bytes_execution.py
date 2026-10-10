"""I execute copied binary host results with exact array origins and grants."""
import json
import os
import subprocess
from pathlib import Path
import unittest
from tests import test_portable_read_execution as read

ROOT = read.ROOT


class PortableBytesExecution(unittest.TestCase):
    setUp = read.PortableReadExecution.setUp
    command = read.PortableReadExecution.command

    portable_flag = '--portable-file-read'
    test_native_allocation_prefixes_and_recovery = read.PortableReadExecution.test_native_allocation_prefixes_and_recovery

    def module(self, body='PUSH_STR path\nCALL_EXTERN 0\nARR_LEN', declaration='.import "" "file_read_bytes" array string'):
        assembly, module = self.work/'bytes.nasm', self.work/'bytes.nvm'
        assembly.write_text(f'{declaration}\n.string path {json.dumps(str(self.data))}\n'
            '.entry main\n.function main 0 2 0 int 1\n'+body+'\nRET\n.end\n')
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        return module

    def execute(self, module, expected=3, host_status=0):
        ir = self.work/'bytes.ll'
        self.command([ROOT/'bin/nvm2llvm', module, '--portable-file-read', '--entry-name', 'nano_entry', '-o', ir])
        host = self.work/'host.c'
        host.write_text(r'''
#include "portable_read_module.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern uint64_t nano_try_entry(void);
extern uint32_t nano_dispose(void);
extern uint64_t nms_module_live_objects(void),nms_module_live_bytes(void);
int main(int argc,char **argv) {
    assert(argc==4);
    NprPath path={(const uint8_t *)argv[1],(uint32_t)strlen(argv[1])};
    NprFileHost *host=NULL;
    assert(npr_file_host_create(&path,1,&host)==NPR_OK);
    NprHostBinding text={npr_file_read,host},bytes={npr_file_read_bytes,host};
    assert(npr_module_bind(&text)==NPR_OK);
    assert((nano_try_entry()>>32)==6 && npr_module_host_status()==NPR_DENIED);
    assert(npr_module_bind_bytes(&bytes)==NPR_OK);
    for(int i=0;i<4;i++) {
        uint64_t actual=nano_try_entry();
        if(actual!=strtoull(argv[2],NULL,10))fprintf(stderr,"result=%llu host=%u\n",(unsigned long long)actual,npr_module_host_status());
        assert(actual==strtoull(argv[2],NULL,10));
        assert(npr_module_host_status()==strtoul(argv[3],NULL,10));
        assert(!nms_module_live_objects() && !nms_module_live_bytes());
    }
    assert(npr_module_bind_bytes(NULL)==NPR_OK);
    assert((nano_try_entry()>>32)==6 && npr_module_host_status()==NPR_DENIED);
    assert(npr_module_bind(NULL)==NPR_OK);
    assert(!nano_dispose());
    assert(!npr_file_host_destroy(host));
    return 0;
}
''')
        native = self.work/'native'
        self.command([os.environ.get('NANO_NATIVE_CLANG','clang'), '-O2', '-std=c11',
                      '-Isrc/nanoisa', ir, host, ROOT/'lib/libnano_portable_read.a', '-o', native])
        self.command([native, self.data, str(expected), str(host_status)])
        if host_status == 0:
            c_source, c_native = self.work/'byte-aot.c', self.work/'byte-aot'
            self.command([ROOT/'bin/nvm2c', module, '-o', c_source])
            self.command([os.environ.get('NANO_NATIVE_CLANG','clang'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                          c_source, '-lm', '-o', c_native])
            ran = subprocess.run([c_native],capture_output=True,text=True,timeout=30)
            self.assertEqual(ran.returncode,expected & 255,ran.stdout+ran.stderr)
        wasm = self.work/'bytes.wasm'
        self.command([ROOT/'bin/nvm2wasm', module, '--portable-file-read', '-o', wasm])
        runner = self.work/'bytes.mjs'
        runner.write_text('import fs from "node:fs";\n'
            f'import {{createFileReadInstance,createReadTextInstance}} from {json.dumps((ROOT/"src/runtime/portable_read_node.mjs").as_uri())};\n'
            'const bytes=fs.readFileSync(process.argv[2]),path=Buffer.from(process.argv[3]);\n'
            'let refused=false;try{createReadTextInstance(bytes,[path]);}catch(e){refused=true;}\n'
            'if(!refused)throw Error("text-only envelope");\n'
            'for(const allowed of [false,true]) {\n'
            'const i=createFileReadInstance(bytes,[path],allowed?[path]:[]);\n'
            'for(let n=0;n<4;n++){const value=i.call("nano_try_entry");\n'
            'if(value!==(allowed?BigInt(process.argv[4]):6n<<32n))throw Error("byte result "+value);\n'
            'if(i.call("npr_module_host_status")!==(allowed?Number(process.argv[5]):1))throw Error("status");}\n'
            'i.call("nano_dispose");i.close();}\n')
        self.command(['node', runner, wasm, self.data, str(expected), str(host_status)])

    def test_binary_aliases_mutation_and_independent_reads(self):
        self.data.write_bytes(b'A\0\xff')
        module = self.module('''PUSH_STR path
CALL_EXTERN 0
STORE_LOCAL 0
LOAD_LOCAL 0
STORE_LOCAL 1
LOAD_LOCAL 1
PUSH_I64 0
PUSH_I64 9
ARR_SET
POP
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
PUSH_U8 9
EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 1
ARR_GET
PUSH_U8 0
EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 2
ARR_GET
PUSH_U8 255
EQ
ASSERT
PUSH_STR path
CALL_EXTERN 0
PUSH_I64 0
ARR_GET
PUSH_U8 65
EQ
ASSERT
LOAD_LOCAL 0
ARR_LEN''')
        self.execute(module)

    def test_text_and_byte_calls_share_workspace_without_sharing_grants(self):
        self.data.write_bytes(b'abc')
        module = self.module('PUSH_STR path\nCALL_EXTERN 0\nSTR_LEN\nSTORE_LOCAL 0\n'
                             'PUSH_STR path\nCALL_EXTERN 1\nARR_LEN\nLOAD_LOCAL 0\nI64_ADD',
                             '.import "" "file_read" string string\n'
                             '.import "" "file_read_bytes" array string')
        self.execute(module, 6)

    def test_empty_missing_and_exact_bound(self):
        module = self.module('PUSH_STR path\nCALL_EXTERN 0\nARR_LEN')
        for content in (b'', None, bytes(range(256))*4096):
            if content is None:
                self.data.unlink()
            else:
                self.data.write_bytes(content)
            self.execute(module, len(content) if content else 0)

    def test_oversize_cleanup(self):
        self.data.write_bytes(b'x'*1048577)
        self.execute(self.module('PUSH_STR path\nCALL_EXTERN 0\nARR_LEN'), 6 << 32, 2)

    def source_products(self, drivers):
        self.data.write_bytes(b'A\0\xff')
        source, module = self.work/'source.nano', self.work/'source.nvm'
        source.write_text('fn main() -> int {\n'
            f'let bytes: array<u8> = (file_read_bytes {json.dumps(str(self.data))})\n'
            'assert (== (array_length bytes) 3)\n'
            'assert (== (at bytes 0) 65)\n'
            'assert (== (at bytes 1) 0)\n'
            'assert (== (at bytes 2) 255)\n'
            'let alias: array<u8> = bytes\n'
            '(array_set alias 0 9)\n'
            'assert (== (at bytes 0) 9)\n'
            f'let fresh: array<u8> = (file_read_bytes {json.dumps(str(self.data))})\n'
            'assert (== (at fresh 0) 65)\nreturn 0\n}\n'
            'shadow main { assert (== (main) 0) }\n')
        for driver in drivers:
            self.command([*driver, source, '--emit-nvm', '-o', module])
            self.command([ROOT/'bin/nano_vm', module])
            c_source, native = self.work/'source.c', self.work/'source-c'
            self.command([ROOT/'bin/nvm2c', module, '-o', c_source])
            self.command([os.environ.get('NANO_NATIVE_CLANG','clang'), c_source,
                          ROOT/'bin/nano_aot_runtime.o', '-lm', '-o', native])
            self.command([native])
            self.execute(module, 0)

    def test_source_products(self):
        self.source_products([[ROOT/'bin/nano_virt'], [ROOT/'bin/nanoc']])

    def test_seed_source_product(self):
        self.source_products([[ROOT/'bin/nano_vm', ROOT/'bin/nanoc_seed.nvm', '--']])

    def test_wrong_array_write_is_refused_before_publication(self):
        module = self.module('PUSH_STR path\nCALL_EXTERN 0\nPUSH_STR path\nARR_PUSH\nARR_LEN')
        out = self.work/'prior.ll'
        out.write_text('prior')
        self.command([ROOT/'bin/nvm2llvm', module, '--portable-file-read', '-o', out],success=False)
        self.assertEqual(out.read_text(),'prior')

    def test_text_only_and_unknown_imports_refuse(self):
        module = self.module('PUSH_STR path\nCALL_EXTERN 0\nARR_LEN')
        out = self.work/'prior.ll'
        out.write_text('prior')
        for flags in ([], ['--portable-read-text']):
            self.command([ROOT/'bin/nvm2llvm', module, *flags, '-o', out], success=False)
            self.assertEqual(out.read_text(),'prior')
        for declaration in ('.import "" "other_read_bytes" array string',
                            '.import "foreign" "file_read_bytes" array string',
                            '.import "" "file_read_bytes" string string'):
            module = self.module('PUSH_STR path\nCALL_EXTERN 0\nPOP\nPUSH_I64 0',declaration)
            self.command([ROOT/'bin/nvm2llvm', module, '--portable-file-read', '-o', out],success=False)
            self.assertEqual(out.read_text(),'prior')


if __name__ == '__main__':
    unittest.main()
