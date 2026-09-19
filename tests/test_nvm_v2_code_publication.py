"""I qualify allocation-failing conversion without executing any module."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_private_owned_array_runtime as support
ROOT = support.ROOT

class CodePublication(unittest.TestCase):
    command = support.PrivateOwnedArrayRuntime.command

    def test_exact_code_and_transient_allocation_refusal(self):
        self.serial = 0
        supplied = os.environ.get('CODE_PUBLICATION_ARTIFACTS')
        self.work = Path(supplied) if supplied else Path(tempfile.mkdtemp(prefix='nano-code-publication-'))
        if supplied:
            self.work.mkdir(parents=True, exist_ok=False)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        flags = shlex.split(os.environ.get('CODE_PUBLICATION_CFLAGS', ''))
        common = [*cc, *flags, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-D_GNU_SOURCE', '-Isrc', '-Isrc/nanoisa']
        hooks = ['-Dmalloc=code_test_malloc', '-Dcalloc=code_test_calloc', '-Drealloc=code_test_realloc', '-Dfree=code_test_free']
        objects = []
        for name in ('nvm_format', 'nvm_v2_convert'):
            obj = self.work / (name + '.o')
            self.command([*common, *hooks, '-c', ROOT/'src/nanoisa'/f'{name}.c', '-o', obj])
            objects.append(obj)
        exe = self.work/'code-publication'
        self.command([*common, ROOT/'tests/nanoisa/test_nvm_v2_code_publication.c', *objects, *shlex.split(os.environ['CODE_PUBLICATION_OBJECTS']), *shlex.split(os.environ['CODE_PUBLICATION_LDFLAGS']), '-o', exe])
        self.assertIn('checked CODE publication controls passed', self.command([exe]).stdout)

if __name__ == '__main__':
    unittest.main()
