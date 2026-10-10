"""I execute declared read-text calls through generated LLVM and Wasm."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PortableReadExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-portable-execution-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.data = self.work / 'data'
        self.data.write_bytes(b'copied')

    def command(self, args, success=True):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode == 0, success, str(args) + '\n' + result.stdout + result.stderr)
        return result

    def module(self, operand='PUSH_STR path', declaration='.import "" "file_read" string string'):
        assembly = self.work / 'input.nasm'
        assembly.write_text(f'{declaration}\n.string path {json.dumps(str(self.data))}\n'
                            '.entry main\n.function main 0 0 0 int 1\n'
                            f'{operand}\nCALL_EXTERN 0\nSTR_LEN\nRET\n.end\n')
        module = self.work / 'input.nvm'
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        return module

    def test_native_binding_and_lifecycle(self):
        module = self.module()
        ir = self.work / 'native.ll'
        self.command([ROOT/'bin/nvm2llvm', module, '--portable-read-text', '--entry-name', 'nano_entry', '-o', ir])
        wrapper = self.work / 'main.c'
        wrapper.write_text(r'''
#include "portable_read_module.h"
#include <assert.h>
#include <string.h>
extern uint64_t nano_try_entry(void);
extern uint32_t nano_dispose(void);
static int calls;
static int fault;
static int32_t read_text(void *context, const uint8_t *path, uint32_t size,
                        uint8_t *out, uint32_t capacity, uint32_t *length) {
    calls++;
    assert(npr_module_bind(NULL)==NPR_INVALID);
    assert(nano_dispose()==4);
    assert((nano_try_entry()>>32)==4);
    if(fault==1)return -1;
    if(fault==2){*length=capacity+1;return NPR_OK;}
    if(fault==3){out[0]=0;*length=1;return NPR_OK;}
    return npr_file_read(context,path,size,out,capacity,length);
}
int main(int argc,char **argv) {
    assert(argc==2);
    assert((nano_try_entry()>>32)==6 && npr_module_host_status()==NPR_DENIED);
    NprPath path={(const uint8_t *)argv[1],(uint32_t)strlen(argv[1])};
    NprFileHost *host=NULL;
    assert(npr_file_host_create(&path,1,&host)==NPR_OK);
    NprHostBinding binding={read_text,host};
    assert(npr_module_bind(&binding)==NPR_OK);
    for(int i=0;i<20;i++)assert(nano_try_entry()==6 && npr_module_host_status()==NPR_OK);
    assert(calls==20);
    for(fault=1;fault<=3;fault++)
        assert((nano_try_entry()>>32)==6 && npr_module_host_status()==NPR_INVALID);
    fault=0;
    assert(nano_try_entry()==6 && npr_module_host_status()==NPR_OK);
    assert(npr_module_bind(NULL)==NPR_OK);
    assert((nano_try_entry()>>32)==6 && npr_module_host_status()==NPR_DENIED);
    assert(calls==24);
    assert(nano_dispose()==0);
    assert(npr_module_bind(&binding)==NPR_INVALID);
    assert((nano_try_entry()>>32)==5);
    assert(npr_file_host_destroy(host)==NPR_OK);
    return 0;
}
''')
        executable = self.work / 'native'
        self.command([os.environ.get('NANO_NATIVE_CLANG', 'clang'), '-O2', '-Wall', '-Wextra', '-Werror',
                      '-Isrc/nanoisa', ir, wrapper, ROOT/'lib/libnano_portable_read.a', '-o', executable])
        self.command([executable, self.data])

    def wasm(self, module):
        output = self.work / 'out.wasm'
        self.command([ROOT/'bin/nvm2wasm', module, '--portable-read-text', '-o', output])
        return output

    def test_wasm_permission_repeated_entry_and_type_guard(self):
        runner = self.work / 'run.mjs'
        runner.write_text('import fs from "node:fs";\n'
            f'import {{createReadTextInstance}} from {json.dumps((ROOT/"src/runtime/portable_read_node.mjs").as_uri())};\n'
            'const bytes=fs.readFileSync(process.argv[2]);let answers=[];\n'
            'for(const paths of [[],[Buffer.from(process.argv[3])]]) {\n'
            'const i=createReadTextInstance(bytes,paths);let rows=[];\n'
            'for(let n=0;n<3;n++)rows.push([String(i.call("nano_try_entry")),i.call("npr_module_host_status")]);\n'
            'i.call("nano_dispose");rows.push([String(i.call("nano_try_entry")),i.call("npr_module_host_status")]);\n'
            'answers.push(rows);i.close();}console.log(JSON.stringify(answers));\n')
        wasm = self.wasm(self.module())
        result = json.loads(self.command(['node', runner, wasm, self.data]).stdout)
        self.assertEqual(result, [[['25769803776',1]]*3+[['21474836480',0]],
                                  [['6',0]]*3+[['21474836480',0]]])
        # General verification is advisory: the emitted tag guard must stop host effects.
        wasm = self.wasm(self.module('PUSH_I64 0'))
        result = json.loads(self.command(['node', runner, wasm, self.data]).stdout)
        self.assertEqual(result, [[['4294967296',0]]*3+[['21474836480',0]]]*2)

    def test_source_vm_c_and_wasm(self):
        source = self.work / 'input.nano'
        source.write_text('extern fn file_read(path: string) -> string\n'
            'fn main() -> int { unsafe {\n'
            f'let text: string = (file_read {json.dumps(str(self.data))})\n'
            'assert (== text "copied")\n} return 0 }\n')
        module = self.work / 'source.nvm'
        self.command([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
        self.command([ROOT/'bin/nano_vm', module])
        c_source, native = self.work/'source.c', self.work/'source-native'
        self.command([ROOT/'bin/nvm2c', module, '-o', c_source])
        self.command([os.environ.get('NANO_NATIVE_CLANG', 'clang'), c_source, '-lm', '-o', native])
        self.command([native])
        ir = self.work/'source.ll'
        self.command([ROOT/'bin/nvm2llvm', module, '--portable-read-text', '--entry-name', 'nano_entry', '-o', ir])
        wrapper = self.work/'source-host.c'
        wrapper.write_text(r'''#include "portable_read_module.h"
#include <assert.h>
#include <string.h>
extern uint64_t nano_try_entry(void);
extern uint32_t nano_dispose(void);
int main(int argc,char **argv) {
    assert(argc==2);
    NprPath path={(const uint8_t *)argv[1],(uint32_t)strlen(argv[1])};
    NprFileHost *host=NULL;
    assert(npr_file_host_create(&path,1,&host)==NPR_OK);
    NprHostBinding binding={npr_file_read,host};
    assert(npr_module_bind(&binding)==NPR_OK);
    assert(nano_try_entry()==0);
    assert(npr_module_bind(NULL)==NPR_OK);
    assert(nano_dispose()==0);
    assert(npr_file_host_destroy(host)==NPR_OK);
    return 0;
}
''')
        llvm_native = self.work/'source-llvm'
        self.command([os.environ.get('NANO_NATIVE_CLANG', 'clang'), '-O2', '-Wall', '-Wextra', '-Werror',
                      '-Isrc/nanoisa', ir, wrapper, ROOT/'lib/libnano_portable_read.a', '-o', llvm_native])
        self.command([llvm_native, self.data])
        wasm = self.wasm(module)
        runner = self.work/'source.mjs'
        runner.write_text('import fs from "node:fs";\n'
            f'import {{createReadTextInstance}} from {json.dumps((ROOT/"src/runtime/portable_read_node.mjs").as_uri())};\n'
            'const i=createReadTextInstance(fs.readFileSync(process.argv[2]),[Buffer.from(process.argv[3])]);\n'
            'if(i.call("nano_try_entry")!==0n)throw Error("source result");\n'
            'i.call("nano_dispose");i.close();\n')
        self.command(['node', runner, wasm, self.data])

    def test_default_refusal_and_exact_imports_preserve_output(self):
        output = self.work / 'output.ll'
        output.write_text('preserved')
        self.command([ROOT/'bin/nvm2llvm', self.module(), '-o', output], success=False)
        self.assertEqual(output.read_text(), 'preserved')
        for declaration in ('.import "other" "file_read" string string',
                            '.import "" "file_read_extra" string string',
                            '.import "" "file_read" int string',
                            '.import "" "file_read" string string\n.import_kind 0 coprocess'):
            self.command([ROOT/'bin/nvm2llvm', self.module(declaration=declaration),
                          '--portable-read-text', '-o', output], success=False)
            self.assertEqual(output.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
