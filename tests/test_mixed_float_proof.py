"""I qualify closed shape evidence without executing pending modules."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class MixedFloatProof(unittest.TestCase):
    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result
    def test_closed_shape_and_atomicity(self):
        with tempfile.TemporaryDirectory(prefix='nano-mixed-proof-') as temp:
            work = Path(temp)
            common = [*shlex.split(os.environ.get('CC', 'cc')),
                      *shlex.split(os.environ.get('MIXED_CFLAGS', '')),
                      '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-Isrc/nanoisa']
            names = ('mixed_float_proof', 'retained_layouts', 'nvm_v2_layouts', 'nvm_v2_cursor')
            objects = []
            for name in names:
                obj = work / (name + '.o'); objects.append(obj)
                self.command([*common, '-Dcalloc=mfp_test_calloc', '-Dmalloc=mfp_test_malloc',
                              '-c', 'src/nanoisa/' + name + '.c', '-o', obj])
            executable = work / 'query'
            self.command([*common, 'tests/nanoisa/test_mixed_float_proof.c', *objects,
                          *shlex.split(os.environ['MIXED_PROOF_LINK_OBJECTS']),
                          '-lm', '-lcrypto', '-o', executable])
            result = self.command([executable])
            self.assertIn('mixed FLOAT proof checks passed; no pending module execution', result.stdout)
            print(result.stdout, end='')
if __name__ == '__main__': unittest.main()
