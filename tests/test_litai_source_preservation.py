"""Real Git custody controls for my incremental adoption."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('preserve', Path(__file__).resolve().parents[1] / 'scripts/litai_source_preservation.py')
preserve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserve)


class SourceCustody(unittest.TestCase):
    def test_longest_prefix_and_ambiguity(self):
        policy = {'owners': {'frontend': ['src/'], 'vm': ['src/nanovm/']}, 'fallback_owner': 'distribution'}
        self.assertEqual(preserve.owner('src/nanovm/vm.c', policy), 'vm')
        self.assertEqual(preserve.owner('src/parser.c', policy), 'frontend')
        self.assertEqual(preserve.owner('README.md', policy), 'distribution')
        policy['owners']['native'] = ['src/nvm2c']
        policy['file_prefixes'] = ['src/nvm2c']
        self.assertEqual(preserve.owner('src/nvm2c.c', policy), 'native')
        policy['owners']['other'] = ['src/nanovm/']
        with self.assertRaises(ValueError):
            preserve.owner('src/nanovm/vm.c', policy)

    def test_real_git_bytes_modes_deletion_and_dirty_source(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True).stdout.decode().strip()
            git('init', '-q')
            git('config', 'user.name', 'Custody fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            (root/'compiler.c').write_bytes(b'int main(void){return 0;}\n')
            (root/'AGENTS.md').write_text('original custom instructions\n')
            git('add', '.')
            git('commit', '-qm', 'original')
            baseline = git('rev-parse', 'HEAD')
            policy = {'baseline_commit': baseline, 'owners': {'frontend': ['compiler.c']},
                      'fallback_owner': 'distribution', 'reviewed_metadata_changes': ['AGENTS.md']}
            self.assertTrue(preserve.inspect(root, policy, 'HEAD')['preserved'])
            (root/'compiler.c').write_text('changed\n')
            report = preserve.inspect(root, policy, 'HEAD')
            self.assertFalse(report['preserved'])
            self.assertEqual(report['dirty_original_paths'], ['compiler.c'])
            git('restore', 'compiler.c')
            git('update-index', '--chmod=+x', 'compiler.c')
            git('commit', '-qm', 'changed executable mode')
            report = preserve.inspect(root, policy, 'HEAD')
            self.assertIn('compiler.c', report['changed_original_paths'])
            git('reset', '--hard', baseline)  # Fixture repository only.
            (root/'AGENTS.md').write_text('reviewed authority delegate\n')
            git('add', '.')
            git('commit', '-qm', 'metadata delegate')
            self.assertTrue(preserve.inspect(root, policy, 'HEAD')['preserved'])
            git('rm', 'compiler.c')
            git('commit', '-qm', 'delete original source')
            self.assertFalse(preserve.inspect(root, policy, 'HEAD')['preserved'])


if __name__ == '__main__':
    unittest.main()
