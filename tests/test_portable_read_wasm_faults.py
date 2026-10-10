"""I qualify generated Wasm host-result allocation failures and recovery."""
import json
import os
from pathlib import Path
import unittest
from tests import test_portable_read_execution as read
from tests import test_portable_bytes_execution as byte

ROOT = read.ROOT


class PortableReadWasmFaults(unittest.TestCase):
    setUp = read.PortableReadExecution.setUp
    command = read.PortableReadExecution.command

    def qualify(self, binary):
        self.data.write_bytes(b'A\0\xff' if binary else b'copied')
        module = (byte.PortableBytesExecution.module(self) if binary else
                  read.PortableReadExecution.module(self))
        ir = self.work/'faults.ll'
        self.command([ROOT/'bin/nvm2llvm', module,
                      '--portable-file-read' if binary else '--portable-read-text',
                      '--entry-name', 'nano_entry', '--runtime-target', 'wasm32', '-o', ir])
        source = ir.read_text()
        signature = 'define internal fastcc ptr @allocate('
        self.assertEqual(source.count(signature), 1)
        # I instrument the actual generated allocator boundary, keeping its
        # freestanding pool and all caller cleanup paths intact.
        source = source.replace(signature, 'define internal fastcc ptr @allocate_real(')
        source += '''
declare i32 @nano_allocation_allowed()
define internal fastcc ptr @allocate(i64 %size) {
entry:
  %allowed = call i32 @nano_allocation_allowed()
  %ok = icmp ne i32 %allowed, 0
  br i1 %ok, label %success, label %failure
success:
  %value = call fastcc ptr @allocate_real(i64 %size)
  ret ptr %value
failure:
  ret ptr null
}
'''
        ir.write_text(source)
        obj = self.work/'faults.o'
        self.command([os.environ.get('NANO_LLC','llc'), '-mtriple=wasm32-unknown-unknown',
                      '-filetype=obj', ir, '-o', obj])
        hooks = self.work/'hooks.c'
        hooks.write_text('''#include <stdint.h>
static int32_t failure = -1;
static uint32_t requests;
void nano_fault(int32_t index) { failure=index; requests=0; }
uint32_t nano_requests(void) { return requests; }
int32_t nano_allocation_allowed(void) { return (int32_t)requests++ != failure; }
''')
        objects = [obj]
        for path in (hooks, ROOT/'src/nanoisa/portable_read_module.c', ROOT/'src/nanoisa/portable_read_wasm.c'):
            target = self.work/(path.stem+'.o')
            self.command([os.environ.get('NANO_WASM_CLANG','clang'), '--target=wasm32-unknown-unknown',
                          '-std=c11', '-O2', '-ffreestanding', '-fno-builtin', '-Wall', '-Wextra', '-Werror',
                          *(['-DNPR_ENABLE_BYTES'] if binary else []), '-Isrc/nanoisa', '-c', path, '-o', target])
            objects.append(target)
        allowed = self.work/'allowed.txt'
        allowed.write_text('npr_wasm_host_read_text\n'+('npr_wasm_host_read_bytes\n' if binary else ''))
        wasm = self.work/'faults.wasm'
        exports = ('nano_try_entry','nano_dispose','npr_module_host_status',
                   'nms_module_live_objects','nms_module_live_bytes','nano_fault','nano_requests')
        self.command([os.environ.get('NANO_WASM_LD','wasm-ld'), '--no-entry', '--fatal-warnings',
                      '--export-memory', '--initial-memory=2097152', '--max-memory=67108864',
                      '-z', 'stack-size=65536', '--allow-undefined-file='+str(allowed),
                      *['--export='+name for name in exports], *objects, '-o', wasm])
        runner = self.work/'faults.mjs'
        runner.write_text('import fs from "node:fs";\n'
            f'import {{createReadTextInstance,createFileReadInstance}} from {json.dumps((ROOT/"src/runtime/portable_read_node.mjs").as_uri())};\n'+'''
const bytes=fs.readFileSync(process.argv[2]), path=Buffer.from(process.argv[3]);
const binary=process.argv[4]==='1', expected=binary?3n:6n;
const make=()=>binary?createFileReadInstance(bytes,[],[path]):createReadTextInstance(bytes,[path]);
const check=(ok,message)=>{if(!ok)throw Error(message);};
const empty=i=>check(i.call('nms_module_live_objects')===0n && i.call('nms_module_live_bytes')===0n,'live managed result');
let i=make();i.call('nano_fault',-1);
check(i.call('nano_try_entry')===expected,'baseline result');
const count=i.call('nano_requests');check(count>0,'allocator not observed');empty(i);
check(i.call('nano_dispose')===0,'baseline dispose');i.close();
const effects=new Set();
for(let prefix=0;prefix<count;prefix++) {
  i=make();i.call('nano_fault',prefix);
  check(i.call('nano_try_entry')===3n<<32n,'allocation prefix '+prefix);
  check(i.call('nano_requests')>prefix,'prefix not reached');
  effects.add(i.report().opened);empty(i);
  i.call('nano_fault',-1);
  check(i.call('nano_try_entry')===expected,'recovery '+prefix);
  check(i.call('npr_module_host_status')===0,'recovery host status');empty(i);
  check(i.call('nano_dispose')===0,'dispose '+prefix);empty(i);i.close();
}
check(effects.has(true),'no post-host failure exercised');
console.log(`I pass ${count} generated Wasm allocation prefixes and recovery; binary=${binary}.`);
''')
        result = self.command(['node', runner, wasm, self.data, '1' if binary else '0'])
        print(result.stdout.strip())

    def test_text_result_allocation_failures(self):
        self.qualify(False)

    def test_byte_result_allocation_failures(self):
        self.qualify(True)


if __name__ == '__main__':
    unittest.main()
