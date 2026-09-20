"""I measure counted mixed storage; I grant no mixed bytecode admission."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock
from scripts import embed_managed_runtime as package
from tests import test_file_cyclic

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/test_mixed_counted_runtime.c'
EXPORTS = ['mixed_counted_all', 'mixed_counted_checks', 'mixed_counted_peak',
           'mixed_counted_fault_cases']


class MixedCountedRuntime(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-mixed-counted-'))
        print(f'I retain counted runtime artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('MC_COUNTED_CC', 'cc'))
        cls.flags = ['-std=c11', '-Wall', '-Wextra', '-Werror', '-g',
                     *shlex.split(os.environ.get('MC_COUNTED_CFLAGS', ''))]
        cls.clang = shlex.split(os.environ.get('NMS_RUNTIME_CLANG', 'clang'))
        cls.opt = shlex.split(os.environ.get('NMS_RUNTIME_OPT', 'opt'))
        cls.native_flags = shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS', ''))
        cls.native_sanitize = shlex.split(os.environ.get('MC_PACKAGE_CFLAGS', ''))
        cls.query_objects = shlex.split(os.environ['RECORD_ARRAY_OBJECTS'])
        cls.query_ldflags = shlex.split(os.environ.get('RECORD_ARRAY_LDFLAGS', '-lm -lcrypto'))
        os.environ['LSAN_OPTIONS'] = ''
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'python': sys.executable, 'cc': cls.cc, 'flags': cls.flags,
            'clang': cls.clang, 'opt': cls.opt, 'native_flags': cls.native_flags,
            'package_fixture_flags': cls.native_sanitize,
            'query_objects': cls.query_objects, 'query_ldflags': cls.query_ldflags,
            'LSAN_OPTIONS': '', 'package_runtime_optimization': 'production O2',
        }, indent=2) + '\n')

    def test_01_query_descriptor_correspondence(self):
        executable = self.artifacts / 'query-catalog'
        self.command('query-build', [*self.cc, *self.flags, '-O1',
            'tests/nanoisa/test_mixed_counted_query.c',
            'tests/nanoisa/record_array_alloc.c', *self.query_objects,
            *self.query_ldflags, '-o', str(executable)])
        output = self.command('query-run', [str(executable), str(self.artifacts)])
        self.assertIn(b'five-tag copied query facts', output)

    def test_02_native_core_and_adapter(self):
        for level in ('0', '2'):
            for observed in (False, True):
                name = f'native-O{level}-' + ('observed' if observed else 'production')
                output = self.artifacts / name
                hooks = ['-DNMS_TESTING', '-DNMS_TEST_ALLOC_HOOKS'] if observed else []
                self.command(name + '-build', [*self.cc, *self.flags, '-O' + level,
                    *hooks, str(FIXTURE), '-o', str(output)])
                text = self.command(name + '-run', [str(output)])
                self.assertIn(b'status 0; no admission', text)
                if observed:
                    self.assertIn(b'position 1 mode 0', text)
                    self.assertIn(b'position 1 mode 1', text)
                    self.assertIn(b'0 live requested bytes', text)
                print(text.decode().strip(), flush=True)

    def wasm_controls(self, name, wasm, observed):
        self.assertEqual(self.command(name + '-wasmtime',
            ['wasmtime', 'run', '--invoke', 'mixed_counted_all', str(wasm)]), b'0\n')
        # Each target instance executes all1024 entries and then terminal disposal.
        script = r'''const fs=require('fs');
const bytes=fs.readFileSync(process.argv[1]);const m=new WebAssembly.Module(bytes);
if(WebAssembly.Module.imports(m).length)throw Error('unexpected imports');
for(let i=0;i<3;i++){
 const e=new WebAssembly.Instance(m).exports;
 const status=e.mixed_counted_all();if(status)throw Error('fixture line '+status);
 const checks=e.mixed_counted_checks(),peak=e.mixed_counted_peak(),faults=e.mixed_counted_fault_cases();
 if(checks<1024n)throw Error('missing assertions');
 if(process.argv[2]==='observed'&&(peak===0n||faults===0n))throw Error('missing fault observation');
 if(process.argv[2]==='production'&&(peak!==0n||faults!==0n))throw Error('unexpected test hooks');
 console.log(JSON.stringify({instance:i,status,checks:String(checks),peak:String(peak),faults:String(faults),imports:[]}));
}'''
        (self.artifacts / (name + '-node.js')).write_text(script)
        self.command(name + '-node', ['node', '-e', script, str(wasm),
            'observed' if observed else 'production'])

    def test_03_wasm_and_production_runtime_package(self):
        target = ['--target=wasm32-unknown-unknown', '-ffreestanding', '-fno-builtin', '-nostdlib']
        link = ['-Wl,--no-entry', '-Wl,--max-memory=4194304',
                *['-Wl,--export=' + name for name in EXPORTS]]
        for level in ('0', '2'):
            for observed in (False, True):
                name = f'wasm-O{level}-' + ('observed' if observed else 'production')
                ir = self.artifacts / (name + '.ll')
                wasm = self.artifacts / (name + '.wasm')
                hooks = ['-DNMS_TESTING', '-DNMS_TEST_ALLOC_HOOKS'] if observed else []
                self.command(name + '-ir', [*self.clang, *target, '-std=c11',
                    '-Wall', '-Wextra', '-Werror', '-O' + level, *hooks,
                    '-S', '-emit-llvm', str(FIXTURE), '-o', str(ir)])
                self.command(name + '-verify', [*self.opt, '-passes=verify', '-disable-output', str(ir)])
                self.command(name + '-link', [*self.clang, *target, '-O' + level,
                    str(ir), *link, '-o', str(wasm)])
                self.wasm_controls(name, wasm, observed)

        # I retain the existing packager's temporary products and every invoke.
        # Only the test harness replaces retention/command transport, not flags.
        outer = self
        class RetainedDirectory:
            def __init__(self, prefix):
                self.path = tempfile.mkdtemp(prefix=prefix, dir=outer.artifacts)
            def __enter__(self):
                return self.path
            def __exit__(self, *_):
                return False
        serial = 0
        def invoke(args, **kwargs):
            nonlocal serial
            self.assertFalse(kwargs)
            serial += 1
            return self.command(f'package-tool-{serial}', list(map(str, args))).decode()
        with mock.patch.object(package.tempfile, 'TemporaryDirectory', RetainedDirectory), \
             mock.patch.object(package, 'invoke', invoke):
            header, manifest, variants = package.generate(self.clang, self.opt)
        (self.artifacts / 'runtime.h').write_text(header)
        (self.artifacts / 'runtime-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        self.assertNotIn('nms_test_', header)
        for path, digest in manifest['sources'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        for variant, data in variants.items():
            ir = self.artifacts / (variant + '-packaged.ll')
            ir.write_text(data['ir'])
            self.assertEqual(hashlib.sha256(ir.read_bytes()).hexdigest(), data['sha256'])
            self.assertNotIn('nms_test_', data['ir'])
            for level in ('0', '2'):
                name = f'packaged-{variant}-O{level}'
                output = self.artifacts / (name + ('.wasm' if variant == 'wasm32' else ''))
                flags = target if variant == 'wasm32' else [*self.native_flags, *self.native_sanitize]
                self.command(name + '-build', [*self.clang, *flags, '-std=c11',
                    '-Wall', '-Wextra', '-Werror', '-O' + level, '-DMC_LINKED',
                    str(FIXTURE), str(ir), *(link if variant == 'wasm32' else []), '-o', str(output)])
                if variant == 'wasm32':
                    self.wasm_controls(name, output, False)
                else:
                    self.assertIn(b'status 0; no admission', self.command(name + '-run', [str(output)]))


if __name__ == '__main__':
    unittest.main()
