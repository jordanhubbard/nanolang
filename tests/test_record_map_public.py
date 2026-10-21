"""Actual public map transport/VM/native boundaries; no source publisher change."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_record_array_vm as array_vm
from tests import test_record_map_native as native
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]


class RecordMapPublic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-record-map-public-'))
        print('I retain actual public map evidence at', cls.work, flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
                     '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror', '-Isrc', '-Isrc/nanoisa']
        cls.objects = shlex.split(os.environ['RECORD_ARRAY_VM_OBJECTS'])
        cls.ldflags = shlex.split(os.environ.get('RECORD_ARRAY_VM_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        cls.modules = shlex.split(os.environ.get('RECORD_MAP_MODULES', ''))
        if len(cls.modules) not in (0, 4, 8, 12):
            raise ValueError('I require zero/four/eight/twelve complete original producer inputs.')
        # Rebuild the real transport ownership closure, including section-codec
        # allocations; retained frontend/common objects keep their attribution.
        cls.sources = list(array_vm.PROVIDERS)
        for obj in cls.objects:
            if '/nanoisa/' in obj:
                source = 'src/nanoisa/' + Path(obj).stem + '.c'
                if (ROOT / source).is_file():
                    cls.sources.append(source)
        cls.sources = list(dict.fromkeys(cls.sources))
        names = {Path(source).stem for source in cls.sources}
        cls.objects = [obj for obj in cls.objects if Path(obj).stem not in names]
        cls.products = {}
        cls.statuses = []
        (cls.work/'selection.json').write_text(json.dumps(dict(cc=cls.cc, flags=cls.flags,
            sources=cls.sources, objects=cls.objects, ldflags=cls.ldflags, modules=cls.modules), indent=2)+'\n')

    def command(self, name, args, build=False):
        p = run_captured(args, cwd=ROOT, env=dict(os.environ, LSAN_OPTIONS='', RECORD_MAP_PUBLIC_OUTPUT=str(self.work)), timeout=180, build=build)
        (self.work/(name+'.stdout')).write_bytes(p.stdout)
        (self.work/(name+'.stderr')).write_bytes(p.stderr)
        self.statuses.append(dict(name=name, command=list(map(str,args)), status=p.returncode))
        (self.work/'status.json').write_text(json.dumps(self.statuses, indent=2)+'\n')
        for bad in (b'AddressSanitizer', b'UndefinedBehaviorSanitizer', b'runtime error:'):
            self.assertNotIn(bad, p.stderr)
        self.assertEqual(p.returncode, 0, p.stderr.decode(errors='replace'))
        self.assertEqual(p.stderr, b'')
        return p.stdout

    def driver(self, goto=False, allocator=False):
        mode = ('alloc' if allocator else 'public') + ('-goto' if goto else '-switch')
        dispatch = [] if goto else ['-DNANO_NO_COMPUTED_GOTO']
        hooks = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if allocator else []
        if mode not in self.products:
            products = []
            for source in self.sources:
                obj = self.work/(mode+'-'+Path(source).stem+'.o')
                self.command(mode+'-'+Path(source).stem, self.cc+self.flags+dispatch+hooks+['-c', source, '-o', str(obj)], True)
                products.append(str(obj))
            tracker = self.work/(mode+'-tracker.o')
            self.command(mode+'-tracker', self.cc+self.flags+['-c','tests/nanoisa/record_array_alloc.c','-o',str(tracker)], True)
            products.append(str(tracker))
            self.products[mode] = products
        source = 'tests/nanoisa/test_record_map_public_alloc.c' if allocator else 'tests/nanoisa/test_record_map_public.c'
        exe = self.work/mode
        self.command(mode+'-harness', self.cc+self.flags+dispatch+hooks+[source]+self.products[mode]+self.objects+self.ldflags+['-o',str(exe)], True)
        return exe

    def test_public_switch_and_native(self):
        output = self.command('public-switch-run', [self.driver(), *self.modules])
        self.assertIn(b'actual public map consumer checks', output)
        print(output.decode(), flush=True)
        for i in range(len(self.modules)):
            original = self.work/f'original-{i}.c'
            for forced in (False, True):
                source = original
                if forced:
                    source = self.work/f'original-{i}-forced.c'
                    source.write_text(native.RecordMapNative.force_collection(original.read_text()))
                for opt in ('0','2'):
                    name=f'native-{i}-{forced}-O{opt}';exe=self.work/name
                    self.command(name+'-build', self.cc+['-std=c11','-g','-O'+opt,'-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all',str(source),'-lm','-o',str(exe)], True)
                    self.command(name+'-run', [exe])

    def test_public_goto(self):
        output = self.command('public-goto-run', [self.driver(goto=True), *self.modules])
        self.assertIn(b'actual public map consumer checks', output)
        print(output.decode(), flush=True)

    def test_public_allocation_prefixes(self):
        output = self.command('public-alloc-run', [self.driver(allocator=True)])
        self.assertIn(b'actual public allocation checks', output)
        self.assertIn(b'public native compiler positions', output)
        print(output.decode(), flush=True)


if __name__ == '__main__':
    unittest.main()
