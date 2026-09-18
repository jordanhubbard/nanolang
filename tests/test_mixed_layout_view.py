"""I qualify private descriptions without executing pending modules."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MixedLayoutView(unittest.TestCase):
    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_descriptions_and_failure_atomicity(self):
        with tempfile.TemporaryDirectory(prefix='nano-mixed-view-') as temp:
            work = Path(temp)
            compiler = shlex.split(os.environ.get('CC', 'cc'))
            flags = shlex.split(os.environ.get('MIXED_CFLAGS', ''))
            common = [*compiler, *flags, '-std=c11', '-O1', '-Wall', '-Wextra',
                      '-Werror', '-Isrc/nanoisa']
            objects = []
            for name in ('retained_layouts', 'nvm_v2_layouts', 'nvm_v2_cursor'):
                obj = work / (name + '.o')
                objects.append(obj)
                self.command([*common, '-Dcalloc=mixed_test_calloc',
                              '-Dmalloc=mixed_test_malloc', '-c',
                              'src/nanoisa/' + name + '.c', '-o', obj])
            executable = work / 'describe'
            self.command([*common, 'tests/nanoisa/test_mixed_layout_view.c', *objects,
                          *shlex.split(os.environ['MIXED_LINK_OBJECTS']),
                          '-lm', '-lcrypto', '-o', executable])
            result = self.command([executable])
            self.assertIn('mixed descriptor checks passed; no module execution', result.stdout)
            print(result.stdout, end='')

if __name__ == '__main__':
    unittest.main()
