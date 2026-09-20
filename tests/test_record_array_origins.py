"""I qualify copied flat-array field facts without runtime admission."""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from tests import test_file_cyclic

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [
    'src/nanoisa/managed_array_shapes.c', 'src/nanoisa/verifier.c',
    'src/nanoisa/verifier_types.c', 'src/nanovm/vm_decode.c',
    'src/nanoisa/ownership_contracts.c', 'src/nanoisa/nvm_v2_layouts.c',
    'src/nanoisa/retained_layouts.c',
]


class RecordArrayOrigins(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-origins-'))
        print(f'I retain query artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_ARRAY_CC', 'cc'))
        cls.flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra',
                     '-Werror', *shlex.split(os.environ.get('RECORD_ARRAY_CFLAGS', ''))]
        names = {Path(p).stem for p in PROVIDERS}
        cls.objects = [p for p in shlex.split(os.environ['RECORD_ARRAY_OBJECTS'])
                       if Path(p).stem not in names]
        cls.ldflags = shlex.split(os.environ.get('RECORD_ARRAY_LDFLAGS', '-lm -lcrypto'))
        os.environ['LSAN_OPTIONS'] = ''
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'python': sys.executable, 'cc': cls.cc, 'flags': cls.flags,
            'objects': cls.objects, 'rebuilt_providers': PROVIDERS,
            'ldflags': cls.ldflags, 'LSAN_OPTIONS': ''}, indent=2) + '\n')

    def test_complete_origin_and_allocation_controls(self):
        for observed in (False, True):
            mode = 'instrumented' if observed else 'linked'
            products = []
            for source in PROVIDERS:
                if observed and Path(source).stem == 'managed_array_shapes':
                    continue  # The fixture includes this TU for internal guards.
                obj = self.artifacts / f'{mode}-{Path(source).stem}.o'
                flags = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []
                self.command(f'{mode}-{Path(source).stem}-build',
                             [*self.cc, *self.flags, *flags, '-c', source, '-o', str(obj)])
                products.append(str(obj))
            exe = self.artifacts / mode
            self.command(mode + '-build', [*self.cc, *self.flags,
                *(['-DRA_WHITEBOX'] if observed else []),
                'tests/nanoisa/test_record_array_origins.c',
                'tests/nanoisa/record_array_alloc.c', *products, *self.objects,
                *self.ldflags, '-o', str(exe)])
            output = self.command(mode + '-run', [str(exe)])
            self.assertIn(b'record-array origin checks; no runtime admission', output)
            if observed:
                self.assertIn(b'every refusal recovered', output)
            print(output.decode().strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
