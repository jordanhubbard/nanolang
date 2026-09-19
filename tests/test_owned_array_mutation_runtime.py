"""I exercise prepared mutation roots with public authority and exact fault sites."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_private_owned_array_runtime as runtime_fixture
ROOT = runtime_fixture.ROOT

class OwnedArrayMutation(unittest.TestCase):
    command = runtime_fixture.PrivateOwnedArrayRuntime.command

    def test_prepared_receivers_and_pack_failure(self):
        self.serial = 0
        supplied = os.environ.get('OWNED_ARRAY_MUTATION_ARTIFACTS')
        self.work = Path(supplied) if supplied else Path(tempfile.mkdtemp(prefix='nano-owner-mutation-'))
        if supplied:
            self.work.mkdir(parents=True, exist_ok=False)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        flags = shlex.split(os.environ.get('OWNED_ARRAY_MUTATION_CFLAGS', ''))
        common = [*cc, *flags, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                  '-D_GNU_SOURCE', '-DNANO_OWNER_ARRAY_PUBLIC_TEST', '-Isrc', '-Isrc/nanoisa']
        providers = shlex.split(os.environ['PRIVATE_OWNER_ARRAY_OBJECTS'])
        libraries = shlex.split(os.environ['PRIVATE_OWNER_ARRAY_LDFLAGS'])
        for threaded in (False, True):
            phase = self.work / ('threaded' if threaded else 'switch')
            phase.mkdir()
            objects = []
            for name, source in [('vm', 'src/nanovm/vm.c'), ('heap', 'src/nanovm/heap.c'), ('nvm2c', 'src/nanoisa/nvm2c.c')]:
                extra = []
                if name == 'vm':
                    extra = ['-DNANO_COMPUTED_GOTO' if threaded else '-DNANO_NO_COMPUTED_GOTO', '-Drealloc=mutation_vm_realloc']
                    macros = self.command([*common, *extra, '-dM', '-E', source]).stdout
                    self.assertIn('#define NANO_COMPUTED_GOTO 1' if threaded else '#define NANO_NO_COMPUTED_GOTO 1', macros)
                    if not threaded:
                        self.assertNotIn('#define NANO_COMPUTED_GOTO ', macros)
                if name == 'heap':
                    extra = ['-Dmalloc=mutation_heap_malloc', '-Dcalloc=mutation_heap_calloc']
                obj = phase / (name + '.o')
                self.command([*common, *extra, '-c', source, '-o', obj])
                objects.append(obj)
            exe = phase / 'mutation'
            self.command([*common, 'tests/nanoisa/test_owned_array_mutation_runtime.c', *objects, *providers, *libraries, '-o', exe])
            self.assertIn('prepared mutation checks passed', self.command([exe, phase]).stdout)
            for case in (0, 1):
                if threaded:
                    self.assertEqual((phase/f'mutation{case}.c').read_bytes(), (self.work/'switch'/f'mutation{case}.c').read_bytes())
                    continue
                source = (phase/f'mutation{case}.c').read_text()
                self.assertIn('a.record=NOWN_ALLOC(1,sizeof(nown_record)+3*sizeof(nown_value));' if case else 'nown_function_1(', source)
                harness = phase/f'native{case}.c'
                harness.write_text((ROOT/'tests/nanoisa/owned_array_mutation_native.inc').read_text().replace('GENERATED', f'mutation{case}.c').replace('PACK_CASE', str(case)))
                for opt in ('-O0', '-O2'):
                    native = phase/f'native{case}{opt}'
                    self.command([*cc, *flags, '-std=c11', '-Wall', '-Wextra', '-Werror', opt, harness, '-o', native])
                    self.command([native])

if __name__ == '__main__':
    unittest.main()
