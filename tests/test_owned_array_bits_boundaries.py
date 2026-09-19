"""I exercise owned-array representation and signed boundaries without widening admission."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_private_owned_array_runtime as runtime_fixture
ROOT = runtime_fixture.ROOT

class OwnedArrayBits(unittest.TestCase):
    command = runtime_fixture.PrivateOwnedArrayRuntime.command

    def test_representation_and_extreme_indices(self):
        self.serial = 0
        supplied = os.environ.get('OWNED_ARRAY_BITS_ARTIFACTS')
        self.work = Path(supplied) if supplied else Path(tempfile.mkdtemp(prefix='nano-owner-bits-'))
        if supplied:
            self.work.mkdir(parents=True, exist_ok=False)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        flags = shlex.split(os.environ.get('OWNED_ARRAY_BITS_CFLAGS', ''))
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
                    extra = ['-DNANO_COMPUTED_GOTO' if threaded else '-DNANO_NO_COMPUTED_GOTO']
                    macros = self.command([*common, *extra, '-dM', '-E', source]).stdout
                    self.assertIn('#define NANO_COMPUTED_GOTO 1' if threaded else '#define NANO_NO_COMPUTED_GOTO 1', macros)
                    if not threaded:
                        self.assertNotIn('#define NANO_COMPUTED_GOTO ', macros)
                if name == 'heap':
                    extra = ['-Dvm_array_get=bits_actual_array_get']
                obj = phase / (name + '.o')
                self.command([*common, *extra, '-c', source, '-o', obj])
                objects.append(obj)
            exe = phase / 'bits'
            self.command([*common, 'tests/nanoisa/test_owned_array_bits_boundaries.c', *objects, *providers, *libraries, '-o', exe])
            self.assertIn('owned ARRAY bit and boundary checks passed', self.command([exe, phase]).stdout)
            for case in range(10):
                if threaded:
                    self.assertEqual((phase/f'bits{case}.c').read_bytes(), (self.work/'switch'/f'bits{case}.c').read_bytes())
                    continue
                source = (phase/f'bits{case}.c').read_text()
                self.assertIn('a.record=NOWN_ALLOC(1,sizeof(nown_record)+3*sizeof(nown_value));', source)
                for helper in (1, 2, 3):
                    self.assertIn(f'static int nown_function_{helper}(nown_value *argument,uint64_t *next_generation,uint64_t generation,nown_value *result,NmsRuntime *managed)', source)
                harness = phase/f'native{case}.c'
                harness.write_text((ROOT/'tests/nanoisa/owned_array_bits_native.inc').read_text().replace('GENERATED', f'bits{case}.c').replace('CASE', str(case)))
                for opt in ('-O0', '-O2'):
                    native = phase/f'native{case}{opt}'
                    self.command([*cc, *flags, '-std=c11', '-Wall', '-Wextra', '-Werror', opt, harness, '-o', native])
                    self.command([native])

if __name__ == '__main__':
    unittest.main()
