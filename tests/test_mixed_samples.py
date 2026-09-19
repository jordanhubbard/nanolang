"""I qualify private composition without executing pending modules."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class MixedSamples(unittest.TestCase):
    def command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result
    def test_private_composition_and_atomicity(self):
        with tempfile.TemporaryDirectory(prefix='nano-mixed-composition-') as temp:
            work = Path(temp)
            common = [*shlex.split(os.environ.get('CC', 'cc')),
                      *shlex.split(os.environ.get('MIXED_CFLAGS', '')),
                      '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-Isrc/nanoisa']
            sources = {'mixed_float_proof': 'shape', 'affine_state': 'facts',
                       'retained_layouts': 'view', 'nvm_v2_layouts': 'layout',
                       'nvm_v2_cursor': 'cursor'}
            objects = []
            for name, domain in sources.items():
                obj = work / (name + '.o'); objects.append(obj)
                self.command([*common, '-Dcalloc=mcs_' + domain + '_calloc',
                              '-Dmalloc=mcs_' + domain + '_malloc', '-Dfree=mcs_test_free',
                              '-c', 'src/nanoisa/' + name + '.c', '-o', obj])
            executable = work / 'query'
            self.command([*common, 'tests/nanoisa/test_mixed_samples.c', *objects,
                          *shlex.split(os.environ['MIXED_SAMPLES_LINK_OBJECTS']),
                          '-lm', '-lcrypto', '-o', executable])
            result = self.command([executable])
            self.assertIn('mixed Samples composition checks passed; no pending module execution', result.stdout)
            print(result.stdout, end='')
if __name__ == '__main__': unittest.main()
