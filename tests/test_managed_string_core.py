"""I test a portable runtime prerequisite without admitting managed opcodes."""
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'src/nanoisa/managed_strings.c'
TEST = ROOT/'tests/nanoisa/test_managed_strings.c'
SMOKE = ROOT/'tests/nanoisa/managed_string_smoke.c'


class ManagedCore(unittest.TestCase):
    def setUp(self):
        self.cc = shlex.split(os.environ.get('CC', 'cc'))
        self.wasm_cc = shlex.split(os.environ.get('NMS_WASM_CC', 'clang'))
        for tool in (self.cc[0], self.wasm_cc[0], 'node', 'wasmtime', 'opt'):
            self.assertIsNotNone(shutil.which(tool), tool)
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-managed-core-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_cmd(self, args):
        result = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                                timeout=60, env={**os.environ,
                                'ASAN_OPTIONS': ('detect_leaks=0' if platform.system() == 'Darwin'
                                                 else 'detect_leaks=1')+':abort_on_error=1'})
        self.assertEqual(result.returncode, 0, str(args)+'\n'+result.stdout+result.stderr)
        return result

    def test_native_core_sanitizers_and_production_mode(self):
        for source, testing in ((TEST, True), (SMOKE, False)):
            exe = self.work/('core' if testing else 'production')
            self.run_cmd(self.cc+['-std=c11','-O1','-g','-Wall','-Wextra','-Werror',
                         '-fsanitize=address,undefined','-fno-sanitize-recover=all']+
                         (['-DNMS_TESTING'] if testing else [])+[CORE,source,'-o',exe])
            self.run_cmd([exe])

    def wasm_build(self, testing):
        out = self.work/('core.wasm' if testing else 'production.wasm')
        names = ['nms_core_tests','nms_failure_tests','nms_reuse_tests','nms_pressure_tests'] if testing else ['nms_product_smoke']
        self.run_cmd(self.wasm_cc+['--target=wasm32-unknown-unknown','-std=c11','-O2',
                     '-ffreestanding','-fno-builtin','-nostdlib','-Wall','-Wextra','-Werror']+
                     (['-DNMS_TESTING'] if testing else [])+
                     [CORE, TEST if testing else SMOKE, '-Wl,--no-entry','-Wl,--fatal-warnings',
                      '-Wl,--max-memory=1048576']+
                     ['-Wl,--export='+name for name in names]+['-o',out])
        return out, names

    def test_wasm_import_free_reuse_pressure_and_fresh_instances(self):
        wasm, names = self.wasm_build(True)
        # The real one-megabyte memory maximum forces memory.grow refusal.
        # Repeated calls reuse one pool; a fresh instance starts independently.
        script = '''const fs=require('fs');
const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if(WebAssembly.Module.imports(m).length) throw Error('unexpected imports');
const names=JSON.parse(process.argv[2]);
for(let instance=0;instance<2;instance++) {
 const e=new WebAssembly.Instance(m).exports;
 for(let round=0;round<3;round++) for(const name of names) {
  const result=e[name](); if(result) throw Error(name+' failed at line '+result);
 }
}
console.log('I passed two instances, three rounds, four runtime groups.');'''
        result = self.run_cmd(['node','-e',script,wasm,json.dumps(names)])
        self.assertIn('two instances', result.stdout)
        for name in names:
            self.assertEqual(self.run_cmd(['wasmtime','run','--invoke',name,wasm]).stdout,'0\n')

    def test_production_wasm_and_verified_runtime_ir_have_no_test_hooks(self):
        wasm, names = self.wasm_build(False)
        self.assertEqual(self.run_cmd(['wasmtime','run','--invoke',names[0],wasm]).stdout,'0\n')
        script = '''const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if(WebAssembly.Module.imports(m).length) throw Error('unexpected imports');
const e=new WebAssembly.Instance(m).exports;
if(e.nms_product_smoke() || e.nms_product_smoke()) throw Error('production core failed');'''
        self.run_cmd(['node','-e',script,wasm])
        ir = self.work/'core.ll'
        self.run_cmd(self.wasm_cc+['--target=wasm32-unknown-unknown','-std=c11','-O2',
                     '-ffreestanding','-fno-builtin','-S','-emit-llvm',CORE,'-o',ir])
        self.assertNotIn('nms_test_', ir.read_text())
        self.assertIn('wasm32', ir.read_text())
        self.run_cmd(['opt','-passes=verify','-disable-output',ir])


if __name__ == '__main__':
    unittest.main()
