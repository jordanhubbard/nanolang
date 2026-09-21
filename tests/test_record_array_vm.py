"""I qualify the distinct private mixed VM; generated/public routes stay closed."""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from tests import test_file_cyclic
from tests import test_record_array_execution

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*test_record_array_execution.PROVIDERS,
    'src/nanovm/vm_dispatch.c', 'src/nanovm/value.c', 'src/nanovm/heap.c',
    'src/nanovm/heap_cycles.c', 'src/nanovm/binding_state.c',
    'src/nanovm/vm.c', 'src/nanovm/vm_callback.c',
    'src/nanovm/vm_ffi.c', 'src/nanovm/vm_builtins.c', 'src/nanovm/cop_protocol.c',
    'src/nanovm/vm_ffi_arrays.c', 'src/runtime/callback_runtime.c',
]


class RecordArrayVm(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-vm-'))
        print(f'I retain private VM artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
            '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra',
            '-Werror', '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME', '-Isrc', '-Isrc/nanoisa']
        names = {Path(source).stem for source in PROVIDERS}
        cls.objects = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_VM_OBJECTS'])
                       if Path(obj).stem not in names]
        cls.ldflags = shlex.split(os.environ.get('RECORD_ARRAY_VM_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        os.environ['LSAN_OPTIONS'] = ''
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'python': sys.executable, 'cc': cls.cc, 'flags': cls.flags,
            'ordinary_common_objects': cls.objects, 'rebuilt_providers': PROVIDERS,
            'ldflags': cls.ldflags, 'LSAN_OPTIONS': '',
            'instrumentation': 'Listed query/VM providers only; common providers retain setup attribution.'},
            indent=2) + '\n')

    def qualify(self, observed):
        for threaded in (False, True):
            mode = ('observed' if observed else 'linked') + ('-goto' if threaded else '-switch')
            dispatch = [] if threaded else ['-DNANO_NO_COMPUTED_GOTO']
            macros = self.command(mode + '-macros',
                [*self.cc, *self.flags, *dispatch, '-dM', '-E', 'src/nanovm/vm.c'])
            if threaded:
                self.assertIn(b'#define NANO_COMPUTED_GOTO 1', macros)
            else:
                self.assertIn(b'#define NANO_NO_COMPUTED_GOTO 1', macros)
                self.assertNotIn(b'#define NANO_COMPUTED_GOTO ', macros)
            products = []
            for source in PROVIDERS:
                if observed and source == 'src/nanovm/vm.c':
                    continue  # My fixture includes the exact VM TU for private controls.
                obj = self.artifacts / f'{mode}-{Path(source).stem}.o'
                hooks = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []
                self.command(f'{mode}-{Path(source).stem}-build',
                    [*self.cc, *self.flags, *dispatch, *hooks, '-c', source, '-o', str(obj)])
                products.append(str(obj))
            exe = self.artifacts / mode
            self.command(mode + '-build', [*self.cc, *self.flags, *dispatch,
                *(['-DVM_RA_WHITEBOX'] if observed else []),
                'tests/nanoisa/test_record_array_vm.c', 'tests/nanoisa/record_array_alloc.c',
                *products, *self.objects, *self.ldflags, '-o', str(exe)])
            output = self.command(mode + '-run', [str(exe)])
            self.assertIn(b'private mixed VM checks; no public or generated consumer admission', output)
            if observed:
                self.assertIn(b'I retired all 93 recipes and checked all 256 opcode decisions', output)
                self.assertIn(b'allocation positions in both modes with independent recovery', output)
            print(output.decode(errors='replace').strip(), flush=True)

    def test_linked_private_vm(self):
        self.qualify(False)

    def test_instrumented_private_vm(self):
        self.qualify(True)


if __name__ == '__main__':
    unittest.main()
