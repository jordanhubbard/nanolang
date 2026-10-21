"""Private map preparation only; no runtime admission."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
from tests import test_record_array_vm

PROVIDERS = test_record_array_vm.PROVIDERS

class RecordMapPreparation(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-map-preparation-'))
        print(f'I retain preparation artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
            '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror',
            '-DNANO_RECORD_MAP_PRIVATE_PREPARATION', '-DNANO_NO_COMPUTED_GOTO', '-Isrc', '-Isrc/nanoisa']
        names = {Path(source).stem for source in PROVIDERS}
        cls.objects = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_VM_OBJECTS'])
                       if Path(obj).stem not in names]
        cls.ldflags = shlex.split(os.environ.get('RECORD_ARRAY_VM_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'cc': cls.cc, 'flags': cls.flags, 'objects': cls.objects,
            'providers': PROVIDERS, 'ldflags': cls.ldflags}, indent=2) + '\n')

    def qualify(self, observed):
        products = []
        mode = "observed" if observed else "linked"
        for source in PROVIDERS:
            if observed and source == 'src/nanovm/vm.c':
                continue
            obj = self.artifacts / (mode + '-' + Path(source).stem + '.o')
            self.command(mode + '-' + Path(source).stem + '-build',
                [*self.cc, *self.flags, *(['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []), '-c', source, '-o', str(obj)])
            products.append(str(obj))
        exe = self.artifacts / (mode + '-preparation')
        self.command(mode + '-harness-build', [*self.cc, *self.flags, *(['-DVM_RM_WHITEBOX'] if observed else []),
            'tests/nanoisa/test_record_map_preparation.c', 'tests/nanoisa/record_array_alloc.c',
            *products, *self.objects, *self.ldflags, '-o', str(exe)])
        output = self.command(mode + '-preparation-run', [str(exe), *shlex.split(os.environ.get('RECORD_MAP_MODULES', ''))])
        self.assertIn(b'private map preparation and independence; execution remains refused', output)
        print(output.decode(), flush=True)

    def test_linked_preparation(self):
        self.qualify(False)

    def test_instrumented_preparation(self):
        self.qualify(True)

if __name__ == '__main__':
    unittest.main()
