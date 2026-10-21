"""Private module-taking native map proof and generated-code qualification."""
import json
import os
from pathlib import Path
import re
import shlex
import signal
import tempfile
import time
import unittest
from tests import test_record_array_vm as array_vm
from tests import test_record_array_execution as execution
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
MACROS = ['-DNANO_RECORD_MAP_PRIVATE_NATIVE', '-DNANO_RECORD_MAP_PRIVATE_PREPARATION',
          '-DNANO_RECORD_MAP_PRIVATE_RUNTIME', '-DNANO_RECORD_MAP_PRIVATE_HEAP',
          '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME', '-DNANO_NO_COMPUTED_GOTO']
CASES = ['alias-replacement', 'missing-void', 'missing-assert', 'unused-declaration',
         'initializer', 'tail-duplicates', 'nonzero-layout', 'growth', 'concat-lifetime', 'concat-empty']


class RecordMapNative(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='nano-record-map-native-'))
        print('I retain private native evidence at', self.work, flush=True)
        self.cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        self.flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
                      '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror', *MACROS]
        self.objects = shlex.split(os.environ['RECORD_ARRAY_VM_OBJECTS'])
        self.ldflags = shlex.split(os.environ.get('RECORD_ARRAY_VM_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        self.modules = shlex.split(os.environ.get('RECORD_MAP_MODULES', ''))
        self.env = dict(os.environ, LSAN_OPTIONS='')
        self.status = []
        (self.work/'selection.json').write_text(json.dumps(dict(cc=self.cc, flags=self.flags,
            objects=self.objects, ldflags=self.ldflags, modules=self.modules), indent=2)+'\n')

    def command(self, name, args, *, build=False, failure=False, assertion=False):
        started = time.monotonic()
        p = run_captured(args, cwd=ROOT, env=self.env, timeout=180, build=build)
        (self.work/(name+'.stdout')).write_bytes(p.stdout)
        (self.work/(name+'.stderr')).write_bytes(p.stderr)
        self.status.append(dict(name=name, argv=args, exit=p.returncode, seconds=time.monotonic()-started))
        (self.work/'status.json').write_text(json.dumps(self.status, indent=2)+'\n')
        self.assertNotIn(b'AddressSanitizer', p.stderr)
        self.assertNotIn(b'UndefinedBehaviorSanitizer', p.stderr)
        self.assertNotIn(b'runtime error:', p.stderr)
        if failure:
            self.assertEqual(p.returncode, -signal.SIGABRT, p.stderr.decode())
            function = rb'nl_main' if assertion else rb'[A-Za-z0-9_]+'
            self.assertRegex(p.stderr, rb'^I stopped at a native invariant in '+function+rb' at generated C line [0-9]+\.\n$')
            self.assertEqual(p.stdout, b'')
        else:
            self.assertEqual(p.returncode, 0, p.stderr.decode())
            self.assertEqual(p.stderr, b'')
        return p.stdout

    def driver(self, allocator=False):
        providers = (execution.PROVIDERS + ['src/nanoisa/nvm2c_callables.c', 'src/nanoisa/nvm2c_shape.c']
                     if allocator else array_vm.PROVIDERS)
        names = {Path(s).stem for s in providers} | {'nvm2c'}
        objects = [p for p in self.objects if Path(p).stem not in names]
        products = []
        for source in providers:
            obj = self.work/(Path(source).stem+'.o')
            wrap = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if allocator else []
            self.command(Path(source).stem+'-build', self.cc+self.flags+wrap+['-c', source, '-o', str(obj)], build=True)
            products.append(str(obj))
        source = 'tests/nanoisa/test_record_map_native_alloc.c' if allocator else 'tests/nanoisa/test_record_map_native.c'
        exe = self.work/'driver'
        flags = [] if allocator else ['-DMAP_NATIVE_WHITEBOX']
        self.command('driver-build', self.cc+self.flags+flags+[source, 'tests/nanoisa/record_array_alloc.c']+
                     products+objects+self.ldflags+['-o', str(exe)], build=True)
        return exe

    def compile_native(self, name, text, opt):
        source = self.work/(name+'.c');source.write_text(text)
        exe = self.work/name
        self.command(name+'-build', self.cc+['-std=c11', '-g', '-O'+opt,
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(source), '-lm', '-o', str(exe)], build=True)
        return exe

    @staticmethod
    def force_collection(text):
        needle = 'if (debt >= nmap_collection_budget) nmap_collect();'
        if text.count(needle) != 1:
            raise AssertionError('I require the exact native collection boundary')
        return text.replace(needle, '(void)debt; nmap_collect();')

    def test_compiler_allocation_prefixes(self):
        exe = self.driver(allocator=True)
        measured = self.command('measure', [str(exe)])
        self.assertRegex(measured, rb'I reached [1-9][0-9]* compiler allocation positions')
        output = self.command('prefixes', [str(exe), 'sweep'])
        self.assertIn(b'transient/persistent plus retry', output)
        print(output.decode(), flush=True)

    def test_native_semantics_and_allocation_prefixes(self):
        exe = self.driver()
        # Required target still exercises synthetic controls without retained external inputs.
        self.assertIn(len(self.modules), (0, 4))
        output = self.command('fixtures', [str(exe), str(self.work), *self.modules])
        self.assertIn(b'private native publication checks', output)
        print(output.decode(), flush=True)
        originals = ['original-'+str(i) for i in range(len(self.modules))]
        for case in originals+CASES:
            text = (self.work/(case+'.c')).read_text()
            clear = ('for(size_t i=0;i<sizeof nglobal/sizeof nglobal[0];i++)nglobal[i]=(nmap_value){0};'
                     if 'static nmap_value nglobal[' in text else '')
            forced = ('#define main fixture_main\n'+self.force_collection(text)+
                '\n#undef main\nint main(void){int r=fixture_main();'+clear+
                'nmap_collect();if(nmap_owned_head||nrec_owned_head||nmap_live_bytes||nagg_live_bytes||nroot_head)abort();return r;}\n')
            for variant, contents in [('raw', text), ('forced', forced)]:
                for opt in ('0', '2'):
                    name=f'{case}-{variant}-O{opt}'
                    native = self.compile_native(name, contents, opt)
                    out = self.command(name+'-run', [str(native)], failure=case=='missing-assert', assertion=True)
                    self.assertEqual(out, b'')
        counts = {}
        for case in originals+['alias-replacement', 'growth', 'concat-lifetime', 'concat-empty']:
            text = (self.work/(case+'.c')).read_text()
            contents = ((ROOT/'tests/nanoisa/native_map_alloc_probe.h').read_text()+
                '\n#define main fixture_main\n'+self.force_collection(text)+
                '\n#undef main\n#undef malloc\n#undef calloc\n#undef realloc\n#undef free\n'
                'int main(int argc,char **argv){if(argc>1)ntest_fail=(size_t)strtoull(argv[1],NULL,10);'
                'if(argc>2)ntest_persistent=1;int r=fixture_main();if(ntest_live||ntest_bytes)abort();'
                'printf("%zu %zu %zu\\n",ntest_calls,ntest_live,ntest_bytes);return r;}\n')
            for opt in ('0', '2'):
                name=f'oom-{case}-O{opt}';native=self.compile_native(name, contents, opt)
                measurement = self.command(name+'-measure', [str(native)])
                count, live, size = map(int, measurement.split())
                self.assertGreater(count, 0);self.assertEqual((live, size), (0, 0));counts[name]=count
                for persistent in (False, True):
                    for position in range(count):
                        self.command(f'{name}-denied-{int(persistent)}-{position}',
                            [str(native), str(position)]+(['persistent'] if persistent else []), failure=True)
                self.assertEqual(self.command(name+'-retry', [str(native)]), measurement)
        (self.work/'native-allocation-counts.json').write_text(json.dumps(counts, indent=2)+'\n')
        print('I covered every measured native allocation prefix in both modes:', counts, flush=True)
