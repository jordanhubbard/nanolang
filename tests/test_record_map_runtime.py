"""Actual private map invocation with copied immutable authority."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
from tests import test_record_array_vm

class RecordMapRuntime(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command
    harness_source = 'tests/nanoisa/test_record_map_runtime.c'
    result_banner = b'actual private map runtime checks; public boundaries checked'

    def test_original_switch(self):
        self.runtime(False)

    def test_original_goto(self):
        self.runtime(True)

    def runtime(self, computed_goto):
        self.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-map-runtime-'))
        print(f'I retain runtime artifacts at {self.artifacts}', flush=True)
        cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
            '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror',
            '-DNANO_RECORD_MAP_PRIVATE_PREPARATION', '-DNANO_RECORD_MAP_PRIVATE_RUNTIME',
            '-DNANO_RECORD_MAP_PRIVATE_HEAP', '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME',
            '-Isrc', '-Isrc/nanoisa']
        if not computed_goto:
            flags.append('-DNANO_NO_COMPUTED_GOTO')
        providers = test_record_array_vm.PROVIDERS
        names = {Path(source).stem for source in providers}
        objects = [p for p in shlex.split(os.environ['RECORD_ARRAY_VM_OBJECTS']) if Path(p).stem not in names]
        ldflags = shlex.split(os.environ.get('RECORD_ARRAY_VM_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        modules = shlex.split(os.environ.get('RECORD_MAP_MODULES', ''))
        (self.artifacts / 'selection.json').write_text(json.dumps({'cc': cc, 'flags': flags,
            'objects': objects, 'ldflags': ldflags, 'modules': modules, 'providers': providers}, indent=2)+'\n')
        macros = self.command('dispatch-macros', [*cc, *flags, '-dM', '-E', 'src/nanovm/vm.c'])
        if computed_goto:
            self.assertIn(b'#define NANO_COMPUTED_GOTO 1', macros)
        else:
            self.assertNotIn(b'#define NANO_COMPUTED_GOTO ', macros)
        products = []
        for source in providers:
            if source=='src/nanovm/vm.c':
                continue
            obj = self.artifacts / (Path(source).stem+'.o')
            self.command(Path(source).stem+'-build', [*cc, *flags, '-DRA_ALLOC_WRAP',
                '-include', 'tests/nanoisa/record_array_alloc.h', '-c', source, '-o', str(obj)])
            products.append(str(obj))
        exe = self.artifacts / 'runtime'
        self.command('harness-build', [*cc, *flags, self.harness_source,
            'tests/nanoisa/record_array_alloc.c', *products, *objects, *ldflags, '-o', str(exe)])
        output = self.command('runtime-run', [str(exe), *modules])
        self.assertIn(self.result_banner, output)
        print(output.decode(), flush=True)

if __name__=='__main__':
    unittest.main()
