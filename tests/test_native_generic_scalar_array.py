"""I pair one generic scalar/array source module across compiler producers."""
from pathlib import Path
import os
import tempfile
import unittest
from tests import test_native_optional_array_reads as base

ROOT = Path(__file__).resolve().parents[1]


class GenericScalarArray(unittest.TestCase):
    checked = base.OptionalArrayReads.checked

    def test_source_producers(self):
        compiler_dir = Path(os.environ.get('NANO_VARIANT_COMPILER_DIR', ROOT/'bin'))
        fixture = ROOT/'tests/nanoisa/fixtures/generic_scalar_array.nano'
        with tempfile.TemporaryDirectory(prefix='generic-scalar-array-') as temp:
            directory = Path(temp)
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=compiler):
                    module = directory/(compiler+'.nvm')
                    source = directory/(compiler+'.c')
                    binary = directory/compiler
                    self.checked([compiler_dir/compiler, fixture, '--emit-nvm', '-o', module])
                    before = module.read_bytes()
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
                    self.checked([ROOT/'bin/nano_vm', module])
                    self.checked([ROOT/'bin/nvm2c', module, '-o', source])
                    self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
                    self.checked([binary])
                    self.assertEqual(module.read_bytes(), before)


if __name__ == '__main__': unittest.main()
