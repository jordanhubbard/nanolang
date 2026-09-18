"""I qualify shared string-array storage without granting opcode admission."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src/nanoisa/managed_strings.c'
FIXTURE = ROOT / 'tests/nanoisa/test_managed_string_arrays.c'

class StringArrayCore(unittest.TestCase):
    def run_command(self, args):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, str(args) + '\n' + result.stdout + result.stderr)
        return result

    def test_native_verified_ir_and_sanitizers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            flags = shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS', ''))
            for testing in (False, True):
                options = ['-DNMS_TESTING'] if testing else []
                ir = path / 'core.ll'
                self.run_command(['clang', *flags, '-std=c11', '-Wall', '-Wextra', '-Werror', '-O1',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                  *options, '-S', '-emit-llvm', CORE, '-o', ir])
                self.run_command(['opt', '-passes=verify', '-disable-output', ir])
                exe = path / 'test'
                self.run_command(['clang', *flags, '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                  *options, ir, FIXTURE, '-o', exe])
                self.run_command([exe])

    def test_wasm_real_targets_reuse_and_instances(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for testing in (False, True):
                options = ['-DNMS_TESTING'] if testing else []
                names = ['nms_array_tests'] + (['nms_array_failures', 'nms_array_reuse', 'nms_array_pressure'] if testing else [])
                ir = path / 'core.ll'
                flags = ['--target=wasm32-unknown-unknown', '-std=c11', '-O2', '-ffreestanding', '-fno-builtin', *options]
                self.run_command(['clang', *flags, '-S', '-emit-llvm', CORE, '-o', ir])
                self.run_command(['opt', '-passes=verify', '-disable-output', ir])
                wasm = path / 'core.wasm'
                self.run_command(['clang', *flags, '-nostdlib', ir, FIXTURE, '-Wl,--no-entry',
                                  '-Wl,--max-memory=1048576', *['-Wl,--export=' + n for n in names], '-o', wasm])
                script = '''const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if(WebAssembly.Module.imports(m).length) throw Error('unexpected import');
for(let i=0;i<2;i++){const e=new WebAssembly.Instance(m).exports;
for(let r=0;r<3;r++) for(const n of JSON.parse(process.argv[2])) {
const result=e[n](); if(result) throw Error(n+': line '+result);}}'''
                self.run_command(['node', '-e', script, wasm, json.dumps(names)])
                for name in names:
                    self.assertEqual(self.run_command(['wasmtime', 'run', '--invoke', name, wasm]).stdout, '0\n')

if __name__ == '__main__':
    unittest.main()
