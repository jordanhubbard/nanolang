"""I qualify private descriptions without executing pending modules."""
import os
import contextlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@contextlib.contextmanager
def workspace():
    target = os.environ.get('OWNED_ARRAY_LAYOUT_ARTIFACTS')
    if target:
        path = Path(target).resolve()
        path.mkdir(parents=True, exist_ok=False)
        yield str(path)
    else:
        with tempfile.TemporaryDirectory(prefix='nano-owned-array-view-') as temporary:
            yield temporary

class OwnedArrayLayouts(unittest.TestCase):
    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        if hasattr(self, 'artifacts'):
            index = self.command_index
            self.command_index += 1
            (self.artifacts / f'{index:02d}-command.json').write_text(json.dumps(list(map(str, args))))
            (self.artifacts / f'{index:02d}-output.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_descriptions_and_failure_atomicity(self):
        with workspace() as temp:
            work = Path(temp)
            self.artifacts = work
            self.command_index = 0
            compiler = shlex.split(os.environ.get('CC', 'cc'))
            flags = shlex.split(os.environ.get('OWNED_ARRAY_LAYOUT_CFLAGS', ''))
            common = [*compiler, *flags, '-std=c11', '-O1', '-Wall', '-Wextra',
                      '-Werror', '-Isrc/nanoisa']
            objects = []
            for name in ('retained_layouts', 'nvm_v2_layouts', 'nvm_v2_cursor'):
                obj = work / (name + '.o')
                objects.append(obj)
                self.command([*common, '-Dcalloc=owned_array_test_calloc',
                              '-Dmalloc=owned_array_test_malloc', '-c',
                              'src/nanoisa/' + name + '.c', '-o', obj])
            executable = work / 'describe'
            self.command([*common, 'tests/nanoisa/test_owned_array_layouts.c', *objects,
                          *shlex.split(os.environ['OWNED_ARRAY_LAYOUT_LINK_OBJECTS']),
                          *shlex.split(os.environ.get('OWNED_ARRAY_LAYOUT_LDFLAGS', '-lm -lcrypto')),
                          '-o', executable])
            result = self.command([executable])
            self.assertIn('owned ARRAY descriptor checks passed; no module execution', result.stdout)
            print(result.stdout, end='')

if __name__ == '__main__':
    unittest.main()
