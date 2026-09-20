"""I retain direct real-engine Wasm host acceptance, never NanoISA admission."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import sys
import tempfile
import unittest
import zipfile
from tests import test_portable_read_adapters

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['src/nanoisa/portable_read_wasm.c', 'src/nanoisa/managed_strings.c',
           'tests/nanoisa/test_portable_read_wasm.c']


class PortableReadWasm(unittest.TestCase):
    dump = test_portable_read_adapters.PortableReadAdapters.dump
    retain = test_portable_read_adapters.PortableReadAdapters.retain
    products = test_portable_read_adapters.PortableReadAdapters.products
    inventory = test_portable_read_adapters.PortableReadAdapters.inventory
    command = test_portable_read_adapters.PortableReadAdapters.command

    def selected(self, name):
        argv = shlex.split(os.environ[name])
        self.assertTrue(argv, name)
        found = shutil.which(argv[0])
        self.assertIsNotNone(found, name)
        argv[0] = str(Path(found).absolute())
        return argv

    def test_real_private_wasm_embeddings(self):
        requested = os.environ.get('PORTABLE_WASM_ARTIFACTS')
        self.artifacts = Path(requested).resolve() if requested else Path(tempfile.mkdtemp(prefix='nano-read-wasm-'))
        if requested:
            self.artifacts.mkdir(parents=True, exist_ok=False)
        print(f'I retain private Wasm reader artifacts at {self.artifacts}', flush=True)
        self.store, self.work = self.artifacts / 'objects', self.artifacts / 'products'
        self.store.mkdir()
        self.work.mkdir()
        self.index = 0
        clang, linker = self.selected('PORTABLE_WASM_CLANG'), self.selected('PORTABLE_WASM_LD')
        node, python = self.selected('PORTABLE_WASM_NODE'), self.selected('PORTABLE_WASM_PYTHON')
        flags = shlex.split(os.environ.get('PORTABLE_WASM_FLAGS', ''))
        self.assertNotIn('-fsanitize=address', flags)  # I do not invent native guest instrumentation.
        manifest_path = ROOT / 'docs/design/portable-read-wasm/wasmtime-43-provenance.json'
        provenance = json.loads(manifest_path.read_text())
        wheel = Path(os.environ['PORTABLE_WASM_WHEEL']).resolve()
        suffix = 'macosx_11_0_arm64.whl' if sys.platform == 'darwin' else 'manylinux2014_aarch64.whl'
        self.assertIn(platform.machine(), ('aarch64', 'arm64'))
        pin = next(row for row in provenance if row['filename'].endswith(suffix))
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), pin['sha256'])
        binding = self.work / 'private-binding'
        binding.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                name = Path(member.filename)
                self.assertFalse(name.is_absolute() or '..' in name.parts)
                self.assertFalse(stat.S_ISLNK(member.external_attr >> 16))
                if member.is_dir():
                    (binding / name).mkdir(parents=True, exist_ok=True)
                else:
                    target = binding / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
        for name, digest in pin['members'].items():
            self.assertEqual(hashlib.sha256((binding / name).read_bytes()).hexdigest(), digest, name)
        self.dump('private-binding-provenance.json', pin)
        self.env = dict(os.environ, PYTHONPATH=str(binding), PYTHONDONTWRITEBYTECODE='1',
                        ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
        self.env.pop('PYTHONOPTIMIZE', None)
        extras = json.loads(os.environ.get('PORTABLE_WASM_EXTRA_TOOLS', '{}'))
        self.assertIsInstance(extras, dict)
        self.assertTrue(all(isinstance(p, str) and Path(p).is_file() for p in extras.values()))
        self.inputs = sorted(set(
            [ROOT / p for p in SOURCES]
            + [p for p in (ROOT / 'src/nanoisa').glob('*.h')]
            + [ROOT / 'src/runtime/portable_read_node.mjs', ROOT / 'src/runtime/portable_read_wasmtime.py',
               ROOT / 'tests/nanoisa/portable_read_wasm_node.mjs', ROOT / 'tests/nanoisa/portable_read_wasm_python.py',
               Path(__file__).resolve(), ROOT / 'tests/test_portable_read_adapters.py',
               ROOT / 'docs/NANOISA_PORTABLE_READ_TEXT_WASM.md', ROOT / 'Makefile.gnu', manifest_path, wheel]
            + [binding / name for name in pin['members']]
            + [Path(a[0]) for a in (clang, linker, node, python)]
            + [Path(p).resolve() for p in extras.values()]))
        before = self.inventory()
        self.dump('inputs-before.json', before)
        self.dump('configuration.json', dict(clang=clang, linker=linker, node=node, python=python,
                                             flags=flags, environment={k: self.env[k] for k in
                                             ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS')}))
        try:
            for label, argv in [('clang',clang),('linker',linker),('node',node),('python',python)]:
                self.command([*argv, '--version'])
            self.command([*python, '-c', 'import importlib.metadata,wasmtime; assert importlib.metadata.version("wasmtime")=="43.0.0"; print(wasmtime.__file__)'])
            vectors = []
            files = self.work / 'files'
            files.mkdir()

            def fnv(data):
                value = 2166136261
                for byte in data:
                    value = ((value ^ byte) * 16777619) & 0xffffffff
                return value

            for name, data in [('empty',b''),('small',b'copied'),('unicode',b'caf\xc3\xa9-\xe2\x82\xac'),
                               ('partial',b'\xf0\x9f'),('nul',b'a\0b'),('exact',b'z'*1048576),
                               ('excess',b'z'*1048577),('missing',None)]:
                path = files / ('caf\u00e9-\u20ac.bin' if name == 'unicode' else name + '.bin')
                if data is not None:
                    path.write_bytes(data)
                status = 2 if name == 'excess' else 0
                expected = b'' if data is None or b'\0' in data or status else data
                vectors.append(dict(name=name,pathHex=os.fsencode(path).hex(),status=status,
                                    length=len(expected),hash=fnv(expected)))
            vector_path = self.work / 'vectors.json'
            vector_path.write_text(json.dumps(dict(vectors=vectors), indent=2)+'\n')
            allowed = self.work / 'allowed-imports.txt'
            allowed.write_text('npr_wasm_host_read_text\n')
            outcomes = []
            for optimization in ('O0','O2'):
                for observed in (False,True):
                    name = optimization + ('-managed-observed' if observed else '-ordinary')
                    objects = []
                    for i, source in enumerate(SOURCES):
                        ir, obj = self.work / f'{name}-{i}.ll', self.work / f'{name}-{i}.o'
                        compile_flags = ['--target=wasm32-unknown-unknown','-std=c11','-'+optimization,
                                         '-ffreestanding','-fno-builtin','-fno-ident','-Wall','-Wextra','-Werror',
                                         '-Isrc/nanoisa',*(['-DNMS_TESTING'] if observed else []),*flags]
                        self.command([*clang,*compile_flags,'-S','-emit-llvm',source,'-o',ir])
                        self.command([*clang,'--target=wasm32-unknown-unknown','-'+optimization,'-Werror',*flags,'-c',ir,'-o',obj])
                        objects.append(obj)
                    guest = self.work / (name + '.wasm')
                    self.command([*linker,'--no-entry','--export-memory','--initial-memory=2097152',
                                  '--max-memory=67108864','-z','stack-size=65536',
                                  '--allow-undefined-file='+str(allowed),*objects,'-o',guest])
                    for engine, argv, fixture, library in (
                        ('Node',node,'tests/nanoisa/portable_read_wasm_node.mjs','src/runtime/portable_read_node.mjs'),
                        ('Wasmtime43',python,'tests/nanoisa/portable_read_wasm_python.py','src/runtime/portable_read_wasmtime.py')):
                        output = self.command([*argv,ROOT/fixture,ROOT/library,guest,vector_path])
                        result = json.loads(output.strip().splitlines()[-1])
                        self.assertEqual(result['engine'],engine)
                        self.assertEqual(result['realVectors'],8)
                        self.assertEqual(result['modeled'],19 if observed else 16)
                        self.assertGreater(result['checks'],100)
                        outcomes.append(dict(guest=name,**result))
                        self.dump('outcomes.json',outcomes)
                        print(json.dumps(outcomes[-1]),flush=True)
            self.assertEqual(len(outcomes),8)
        finally:
            after = self.inventory()
            self.dump('inputs-after.json', after)
            self.dump('phase-summary.json', dict(commands=self.index,inputs_equal=before==after,
                      instrumentation='NMS_TESTING guest core only; host fault hooks separate; engines not sanitizer rebuilt'))
            self.assertEqual(before,after)


if __name__ == '__main__':
    unittest.main()
