"""I retain constructor invariants without executing bytecode."""
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from tests import test_portable_read_adapters as retained

ROOT = Path(__file__).resolve().parents[1]

class AffinePrivateVariants(unittest.TestCase):
    dump = retained.PortableReadAdapters.dump
    retain = retained.PortableReadAdapters.retain
    products = retained.PortableReadAdapters.products
    command = retained.PortableReadAdapters.command

    def test_constructors_and_allocation_failures(self):
        supplied = os.environ.get('AFFINE_VARIANTS_ARTIFACTS')
        self.artifacts = Path(supplied).resolve() if supplied else Path(tempfile.mkdtemp(prefix='nano-affine-variants-'))
        if supplied:
            self.artifacts.mkdir(parents=True, exist_ok=False)
        self.store = self.artifacts / 'objects'
        self.work = self.artifacts / 'products'
        self.store.mkdir()
        self.work.mkdir()
        self.index = 0
        self.env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
        cc = shlex.split(os.environ.get('AFFINE_VARIANTS_CC', 'cc'))
        self.assertTrue(cc)
        selected = shutil.which(cc[0])
        self.assertIsNotNone(selected)
        cc[0] = str(Path(selected).resolve())
        flags = shlex.split(os.environ.get('AFFINE_VARIANTS_CFLAGS', ''))
        objects = shlex.split(os.environ['AFFINE_VARIANTS_OBJECTS'])
        self.assertFalse(any(Path(p).name in ('affine_state.o', 'nvm_v2_layouts.o') for p in objects))
        libraries = shlex.split(os.environ['AFFINE_VARIANTS_LDFLAGS'])
        inputs = [Path(cc[0]), *(ROOT / p for p in objects), Path(__file__), Path(retained.__file__),
                  ROOT / 'tests/nanoisa/test_affine_private_variants.c',
                  *sorted((ROOT / 'src/nanoisa').glob('*.h')),
                  *sorted((ROOT / 'src/nanoisa').glob('*.inc')),
                  ROOT / 'src/nanoisa/affine_state.c', ROOT / 'src/nanoisa/nvm_v2_layouts.c']
        inventory = lambda: {str(p.resolve()): self.retain(p) for p in inputs}
        before = inventory()
        self.dump('inputs-before.json', before)
        try:
            common = [*cc, *flags, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-UNDEBUG', '-Isrc/nanoisa']
            decoder = self.work / 'nvm_v2_layouts.o'
            self.command([*common, '-Dmalloc=variants_test_malloc', '-Dcalloc=variants_test_calloc',
                          '-Dfree=variants_test_free', '-c', 'src/nanoisa/nvm_v2_layouts.c', '-o', decoder])
            executable = self.work / 'constructors'
            self.command([*common, 'tests/nanoisa/test_affine_private_variants.c', decoder,
                          *objects, *libraries, '-o', executable])
            self.assertIn('private affine variant checks passed; no bytecode execution',
                          self.command([executable]))
        finally:
            after = inventory()
            self.dump('inputs-after.json', after)
            self.assertEqual(before, after)

if __name__ == '__main__':
    unittest.main()
