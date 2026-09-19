"""I retain old owner-field aliases across distinct ARRAY-local replacement."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_private_owned_array_runtime as runtime_fixture
ROOT = runtime_fixture.ROOT


class OwnedArrayOverwrite(unittest.TestCase):
    command = runtime_fixture.PrivateOwnedArrayRuntime.command

    def test_distinct_local_replacement_and_terminal_roots(self):
        self.serial = 0
        supplied = os.environ.get('OWNED_ARRAY_OVERWRITE_ARTIFACTS')
        self.work = Path(supplied) if supplied else Path(tempfile.mkdtemp(prefix='nano-owner-overwrite-'))
        if supplied:
            self.work.mkdir(parents=True, exist_ok=False)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        flags = shlex.split(os.environ.get('OWNED_ARRAY_OVERWRITE_CFLAGS', ''))
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
                    extra = ['-Dmalloc=overwrite_malloc', '-Dcalloc=overwrite_calloc', '-Drealloc=overwrite_realloc']
                obj = phase / (name + '.o')
                self.command([*common, *extra, '-c', source, '-o', obj])
                objects.append(obj)
            exe = phase / 'overwrite'
            self.command([*common, 'tests/nanoisa/test_owned_array_overwrite.c', *objects, *providers, *libraries, '-o', exe])
            self.assertIn('distinct-array overwrite checks passed', self.command([exe, phase]).stdout)
            for case in (0, 1):
                if threaded:
                    self.assertEqual((phase / f'overwrite{case}.c').read_bytes(), (self.work / 'switch' / f'overwrite{case}.c').read_bytes())
                    continue
                source = (phase / f'overwrite{case}.c').read_text()
                # I require the existing pre-disposal root check; status3 must
                # never pass as success, assertion2 or injected allocation1.
                self.assertIn('if(managed->live_objects || managed->live_bytes)status=3;', source)
                harness = phase / f'native{case}.c'
                harness.write_text((ROOT / 'tests/nanoisa/owned_array_overwrite_native.inc').read_text()
                                   .replace('GENERATED', f'overwrite{case}.c').replace('ASSERTION_CASE', str(case)))
                for opt in ('-O0', '-O2'):
                    native = phase / f'native{case}{opt}'
                    self.command([*cc, *flags, '-std=c11', '-Wall', '-Wextra', '-Werror', opt, harness, '-o', native])
                    self.command([native])


if __name__ == '__main__':
    unittest.main()
