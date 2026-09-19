"""I qualify private origin summaries without executing pending modules."""
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
    target = os.environ.get('OWNED_ARRAY_AUTHORITY_ARTIFACTS')
    if target:
        path = Path(target).resolve()
        path.mkdir(parents=True, exist_ok=False)
        yield str(path)
    else:
        with tempfile.TemporaryDirectory(prefix='nano-owned-array-view-') as temporary:
            yield temporary

class OwnedArrayAuthority(unittest.TestCase):
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

    def test_authority_and_failure_atomicity(self):
        with workspace() as temp:
            work = Path(temp)
            self.artifacts = work
            self.command_index = 0
            compiler = shlex.split(os.environ.get('CC', 'cc'))
            flags = shlex.split(os.environ.get('OWNED_ARRAY_AUTHORITY_CFLAGS', ''))
            common = [*compiler, *flags, '-std=c11', '-O1', '-Wall', '-Wextra',
                      '-Werror', '-Isrc/nanoisa']
            objects = []
            for name in ('retained_layouts', 'nvm_v2_layouts', 'nvm_v2_cursor', 'affine_state', 'verifier'):
                obj = work / (name + '.o')
                objects.append(obj)
                boundary = ['-Dnvm_retained_layouts_valid=authority_structural_layout_valid'] if name == 'verifier' else []
                self.command([*common, *boundary, '-Dcalloc=authority_test_calloc',
                              '-Dmalloc=authority_test_malloc', '-c',
                              'src/nanoisa/' + name + '.c', '-o', obj])
            executable = work / 'describe'
            self.command([*common, 'tests/nanoisa/test_owned_array_authority.c', *objects,
                          *shlex.split(os.environ['OWNED_ARRAY_AUTHORITY_LINK_OBJECTS']),
                          '-lm', '-lcrypto', '-o', executable])
            result = self.command([executable])
            self.assertIn('private owner ARRAY authority checks passed; no pending module execution', result.stdout)
            print(result.stdout, end='')

if __name__ == '__main__':
    unittest.main()
