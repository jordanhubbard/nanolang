"""I execute the exact collection artifact ABI and emitted snapshot failure paths."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
SIGNATURES = {
    'nl_hm_new': 'opaque', 'nl_hm_put': 'void opaque string string',
    'nl_hm_has': 'bool opaque string', 'nl_hm_get': 'string opaque string',
    'nl_hm_size': 'int opaque', 'nl_hm_keys': 'array opaque', 'nl_hm_values': 'array opaque',
    'nl_hm_remove': 'void opaque string', 'nl_hm_clear': 'void opaque', 'nl_hm_free': 'void opaque',
    'nl_set_new': 'opaque', 'nl_set_add': 'void opaque string', 'nl_set_has': 'bool opaque string',
    'nl_set_size': 'int opaque', 'nl_set_values': 'array opaque', 'nl_set_remove': 'void opaque string',
    'nl_set_clear': 'void opaque', 'nl_set_free': 'void opaque',
}
SANITIZERS = ['-fsanitize=address,undefined', '-fno-sanitize-recover=all']
RUNTIME = ['src/runtime/dyn_array.c', 'src/runtime/gc.c', 'src/runtime/gc_struct.c']


class CollectionAot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='collection-aot-', dir=ROOT/'.test_output'))
        print('I retain actual collection AOT evidence at', cls.work, flush=True)
        cls.operations, cls.products = [], {}
        cls.cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC', 'cc'))
        cls.tools = {str(ROOT/'bin'/name): hashlib.sha256((ROOT/'bin'/name).read_bytes()).hexdigest()
                     for name in ('nanoisa', 'nano_vm', 'nvm2c')}
        launcher = str(Path(shutil.which(cls.cc[0]) or cls.cc[0]).resolve(strict=True))
        cls.tools[launcher] = hashlib.sha256(Path(launcher).read_bytes()).hexdigest()
        cls.inputs = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                      ['tests/test_collection_aot.py', 'tests/nanoisa/test_collection_snapshot_aot.c', 'src/nanoisa/nvm2c.c',
                       'modules/std/collections/collections.c', 'modules/std/collections/collections.h',
                       'src/runtime/dyn_array.h', 'modules/std/json/json.c', 'modules/std/fs.c',
                       'src/cJSON.c', 'src/utf8.c', *RUNTIME]}

    @classmethod
    def tearDownClass(cls):
        assert cls.tools == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cls.tools}
        assert cls.inputs == {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in cls.inputs}
        (cls.work/'inputs-tools-before-after.json').write_text(json.dumps({'tools': cls.tools, 'inputs': cls.inputs}, indent=2)+'\n')
        (cls.work/'products.json').write_text(json.dumps(cls.products, indent=2)+'\n')

    def command(self, label, argv, expected=0, env=None):
        label = self._testMethodName+'-'+label
        argv = list(map(str, argv))
        result = run_captured(argv, cwd=ROOT, env=env, timeout=90)
        (self.work/(label+'.stdout')).write_bytes(result.stdout)
        (self.work/(label+'.stderr')).write_bytes(result.stderr)
        self.operations.append({'label': label, 'argv': argv, 'returncode': result.returncode})
        (self.work/'operations.json').write_text(json.dumps(self.operations, indent=2)+'\n')
        if expected is None:
            self.assertNotEqual(result.returncode, 0, label)
        else:
            self.assertEqual(result.returncode, expected, result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))
        return result

    def native_product(self, path, shared=False):
        data = path.read_bytes()
        if data[:4] == b'\xcf\xfa\xed\xfe':
            self.assertEqual(struct.unpack_from('<I', data, 12)[0], 6 if shared else 2)
        else:
            self.assertEqual(data[:4], b'\x7fELF')
            kind = struct.unpack_from('<H' if data[5] == 1 else '>H', data, 16)[0]
            self.assertIn(kind, (3,) if shared else (2, 3))
        self.products[str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'shared': shared}

    def run_native(self, label, path, **kwargs):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = self.command(label, [path], **kwargs)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.native_product(path)
        return result

    def library(self, name, sources, sanitized=False, extra=()):
        path = self.work/(name+('.dylib' if sys.platform == 'darwin' else '.so'))
        self.command(name, [*self.cc, '-std=c99', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                     '-UNDEBUG', '-pthread', '-Isrc', *(['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']),
                     *(SANITIZERS if sanitized else []), *sources, *extra, '-lm', '-o', path])
        self.native_product(path, shared=True)
        return path

    def module(self, label, text):
        source, module = self.work/(label+'.nasm'), self.work/(label+'.nvm')
        source.write_text(text)
        self.command(label+'-assemble', [ROOT/'bin/nanoisa', 'asm', source, '-o', module])
        self.command(label+'-verify', [ROOT/'bin/nano_vm', '--verify-only', module])
        return module

    def translate(self, label, module):
        path = self.work/(label+'.c')
        self.command(label+'-translate', [ROOT/'bin/nvm2c', module, '-o', path])
        return path

    def build(self, label, source, optimization, extra=()):
        path = self.work/(label+optimization)
        self.command(label+'-build'+optimization, [*self.cc, '-std=c11', optimization, '-g', '-Wall',
                     '-Wextra', '-Werror', '-UNDEBUG', '-pthread', '-Isrc', *SANITIZERS,
                     source, *extra, ROOT/'src/utf8.c', '-lm', *(['-ldl'] if sys.platform.startswith('linux') else []), '-o', path])
        data = path.read_bytes()
        self.assertIn(b'__asan_init', data)
        self.assertIn(b'__ubsan_handle', data)
        return path

    def imports(self, library, signatures=SIGNATURES):
        return ''.join(f'.import {json.dumps(str(library))} {json.dumps(name)} {signature}\n.import_kind {i} artifact\n'
                       for i, (name, signature) in enumerate(signatures.items()))

    def program(self, library):
        indices = {name: i for i, name in enumerate(SIGNATURES)}
        def call(name): return f'CALL_EXTERN {indices[name]}\n'
        def check_int(value): return f'PUSH_I64 {value}\nEQ\nASSERT\n'
        text = self.imports(library)+''.join(f'.string k{i} "k{i}"\n.string v{i} "{i*2}"\n' for i in range(200))
        text += '.string d "d"\n.string aa "aa"\n.string one "1"\n.string two "2"\n'
        body = call('nl_hm_new')+'STORE_LOCAL 0\n'
        for key, value in [('d', 'one'), ('aa', 'two')]:
            body += 'LOAD_LOCAL 0\n'+f'PUSH_STR {key}\nPUSH_STR {value}\n'+call('nl_hm_put')
        for key, value in [('d', 'one'), ('aa', 'two')]:
            body += f'LOAD_LOCAL 0\nPUSH_STR {key}\n'+call('nl_hm_get')+f'PUSH_STR {value}\nEQ\nASSERT\n'
        body += 'LOAD_LOCAL 0\nPUSH_STR d\n'+call('nl_hm_remove')
        body += 'LOAD_LOCAL 0\nPUSH_STR d\n'+call('nl_hm_has')+'NOT\nASSERT\n'
        body += 'LOAD_LOCAL 0\nPUSH_STR aa\n'+call('nl_hm_has')+'ASSERT\n'
        for i in range(200):
            body += f'LOAD_LOCAL 0\nPUSH_STR k{i}\nPUSH_STR v{i}\n'+call('nl_hm_put')
        body += 'LOAD_LOCAL 0\n'+call('nl_hm_size')+check_int(201)
        for i in range(100):
            body += f'LOAD_LOCAL 0\nPUSH_STR k{i}\n'+call('nl_hm_remove')
        body += 'LOAD_LOCAL 0\n'+call('nl_hm_size')+check_int(101)
        for name, local in [('nl_hm_keys', 1), ('nl_hm_values', 2), ('nl_hm_keys', 3)]:
            body += 'LOAD_LOCAL 0\n'+call(name)+f'STORE_LOCAL {local}\nLOAD_LOCAL {local}\nARR_LEN\n'+check_int(101)
        for i in range(101):
            body += f'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 {i}\nARR_GET\n'+call('nl_hm_get')
            body += f'LOAD_LOCAL 2\nPUSH_I64 {i}\nARR_GET\nEQ\nASSERT\n'
        body += 'LOAD_LOCAL 0\n'+call('nl_hm_clear')+'LOAD_LOCAL 0\n'+call('nl_hm_size')+check_int(0)
        body += 'LOAD_LOCAL 0\n'+call('nl_hm_free')+'LOAD_LOCAL 0\n'+call('nl_hm_free')
        for i in range(101):
            body += f'LOAD_LOCAL 1\nPUSH_I64 {i}\nARR_GET\nLOAD_LOCAL 3\nPUSH_I64 {i}\nARR_GET\nEQ\nASSERT\n'
        body += call('nl_set_new')+'STORE_LOCAL 4\n'
        for key in ('d', 'aa', 'd', 'k0'):
            body += f'LOAD_LOCAL 4\nPUSH_STR {key}\n'+call('nl_set_add')
        body += 'LOAD_LOCAL 4\n'+call('nl_set_size')+check_int(3)
        body += 'LOAD_LOCAL 4\nPUSH_STR d\n'+call('nl_set_remove')
        body += 'LOAD_LOCAL 4\nPUSH_STR d\n'+call('nl_set_has')+'NOT\nASSERT\n'
        body += 'LOAD_LOCAL 4\nPUSH_STR aa\n'+call('nl_set_has')+'ASSERT\n'
        body += 'LOAD_LOCAL 4\n'+call('nl_set_values')+'STORE_LOCAL 5\n'
        body += 'LOAD_LOCAL 4\n'+call('nl_set_clear')+'LOAD_LOCAL 4\n'+call('nl_set_values')+'ARR_LEN\n'+check_int(0)
        body += 'LOAD_LOCAL 4\n'+call('nl_set_free')+'LOAD_LOCAL 4\n'+call('nl_set_size')+check_int(0)
        body += 'LOAD_LOCAL 5\nARR_LEN\n'+check_int(2)
        # I retain nonzero opaque identities after retirement and refuse wrong-kind use.
        body += 'LOAD_LOCAL 4\n'+call('nl_hm_size')+check_int(0)
        body += 'LOAD_LOCAL 0\n'+call('nl_set_size')+check_int(0)
        return text+'.entry 0\n.function main 0 6 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n'

    def test_real_collection_vm_and_native(self):
        source = 'modules/std/collections/collections.c'
        plain = self.library('collections-plain', [source, *RUNTIME])
        module = self.module('all-vm', self.program(plain))
        self.command('all-vm', [ROOT/'bin/nano_vm', module])
        library = self.library('collections-sanitized', [source, *RUNTIME], sanitized=True)
        module = self.module('all-native', self.program(library))
        generated = self.translate('all-native', module)
        for optimization in ('-O0', '-O2'):
            self.run_native('all-run'+optimization, self.build('all-native', generated, optimization))

    def test_all_malformed_signatures_preserve_output(self):
        for name, signature in SIGNATURES.items():
            fields = signature.split()
            changes = [('result', ['int' if fields[0] != 'int' else 'string', *fields[1:]]),
                       ('arity', [*fields, 'int'])]
            if len(fields) > 1: changes.append(('parameter', [fields[0], 'int', *fields[2:]]))
            for kind, bad in changes:
                label = name+'-'+kind
                module = self.module(label, self.imports(Path('/absent/never-loaded.so'), {name: ' '.join(bad)})+
                                     '.entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                target = self.work/(label+'.c'); target.write_text('I retain the prior product.\n')
                result = self.command(label+'-refuse', [ROOT/'bin/nvm2c', module, '-o', target], expected=None)
                self.assertIn(b'typed value adapter', result.stderr)
                self.assertEqual(target.read_text(), 'I retain the prior product.\n')

    def test_actual_emitted_copy_failures(self):
        module = self.module('copy-control', self.imports(Path('/never-loaded.so'), {'nl_hm_keys': 'array opaque'})+
                             '.entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
        generated = self.translate('copy-control', module)
        original = generated.read_text()
        begin = original.index('static inline nsarr_t nhost_collection_snapshot(')
        end = original.index('static inline nsarr_t nhost_walk_', begin)
        prefix = '#define calloc probe_calloc\n#define malloc probe_malloc\n#define free probe_free\n'
        suffix = '#undef calloc\n#undef malloc\n#undef free\n'
        body = original[begin:end]
        transformed = original[:begin]+prefix+body+suffix+original[end:]
        self.assertEqual(transformed.replace(prefix, '', 1).replace(suffix, '', 1), original)
        self.assertEqual(transformed.count('abort(); } while (0)'), 1)
        transformed = transformed.replace('abort(); } while (0)', 'probe_abort(); } while (0)', 1)
        observed = self.work/'copy-observed.c'; observed.write_text(transformed)
        extra = ['-DCOLLECTION_AOT_PROGRAM='+json.dumps(str(observed)), 'modules/std/collections/collections.c', *RUNTIME]
        for optimization in ('-O0', '-O2'):
            binary = self.build('copy-control', ROOT/'tests/nanoisa/test_collection_snapshot_aot.c', optimization, extra)
            result = self.run_native('copy-control-run'+optimization, binary)
            self.assertIn(b'seven allocation denials', result.stdout)

    def test_companion_and_marker_origin_refusals(self):
        for case in ('both-dependency', 'companion-dependency', 'marker-dependency', 'bad-marker', 'missing-companion', 'missing-marker'):
            marker = self.work/(case+'.called')
            touch = 'FILE *f = fopen('+json.dumps(str(marker))+', "a"); if (f) fclose(f); abort();'
            dependency = self.work/(case+'-dependency.c')
            dependency.write_text('#include <stdbool.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n'
                'int dependency_anchor(void) { return 1; }\n'+
                ('const uint32_t nl_hm_keys__nano_array_abi = 1;\n' if case in ('both-dependency', 'marker-dependency') else '')+
                ('bool nl_hm_keys__nano_array_release_v1(void *v) { (void)v; '+touch+' }\n'
                 if case in ('both-dependency', 'companion-dependency') else ''))
            dep = self.library(case+'-dependency', [dependency])
            origin = self.work/(case+'-origin.c')
            origin.write_text('#include <stdbool.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n'
                'extern int dependency_anchor(void);\nvoid *nl_hm_new(void) { return NULL; }\n'
                'void *nl_hm_keys(void *v) { (void)v; (void)dependency_anchor(); '+touch+' }\n'+
                ('const uint32_t nl_hm_keys__nano_array_abi = '+('2' if case == 'bad-marker' else '1')+';\n'
                 if case not in ('both-dependency', 'marker-dependency', 'missing-marker') else '')+
                ('bool nl_hm_keys__nano_array_release_v1(void *v) { (void)v; '+touch+' }\n'
                 if case not in ('both-dependency', 'companion-dependency', 'missing-companion') else ''))
            library = self.library(case+'-origin', [origin, dep], extra=['-Wl,-rpath,'+str(self.work)])
            text = self.imports(library, {'nl_hm_new': 'opaque', 'nl_hm_keys': 'array opaque'})
            text += '.entry 0\n.function main 0 0 0 int 1\nCALL_EXTERN 0\nCALL_EXTERN 1\nPOP\nPUSH_I64 0\nRET\n.end\n'
            generated = self.translate(case, self.module(case, text))
            for optimization in ('-O0', '-O2'):
                result = self.run_native(case+'-refuse'+optimization, self.build(case, generated, optimization), expected=None)
                self.assertEqual(result.returncode, -signal.SIGABRT)
                self.assertIn(b'I stopped at a native invariant', result.stderr)
                self.assertNotIn(b'ERROR: AddressSanitizer', result.stderr)
                self.assertFalse(marker.exists(), 'I refuse origin errors before calling the producer or cleanup.')

    def test_real_json_and_filesystem_array_types(self):
        json_lib = self.library('json-sanitized', ['modules/std/json/json.c', 'src/cJSON.c', *RUNTIME], sanitized=True)
        fs_lib = self.library('fs-sanitized', ['modules/std/fs.c', *RUNTIME], sanitized=True)
        directory = self.work/'walk'; directory.mkdir(); child = directory/'entry'; child.write_text('I retain one entry.\n')
        entries = [(json_lib, 'nl_json_parse', 'opaque string'),
                   (json_lib, 'nl_json_object_keys', 'array opaque'),
                   (json_lib, 'nl_json_free', 'void opaque'),
                   (fs_lib, 'fs_walkdir', 'array string')]
        text = ''.join(f'.import {json.dumps(str(lib))} {json.dumps(name)} {signature}\n.import_kind {i} artifact\n'
                       for i, (lib, name, signature) in enumerate(entries))
        text += '.string json '+json.dumps('{"key":1}')+'\n.string key "key"\n'
        text += '.string root '+json.dumps(str(directory))+'\n.string path '+json.dumps(str(child))+'\n'
        text += ('.entry 0\n.function main 0 3 0 int 1\nPUSH_STR json\nCALL_EXTERN 0\nSTORE_LOCAL 0\n'
                 'LOAD_LOCAL 0\nCALL_EXTERN 1\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nCALL_EXTERN 2\n'
                 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_STR key\nEQ\nASSERT\n'
                 'PUSH_STR root\nCALL_EXTERN 3\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n'
                 'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nPUSH_STR path\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        generated = self.translate('json-fs', self.module('json-fs', text))
        for optimization in ('-O0', '-O2'):
            self.run_native('json-fs-run'+optimization, self.build('json-fs', generated, optimization))


if __name__ == '__main__':
    unittest.main()
