"""I check that the self-hosted compiler lowers the std/regex C-backed externs and the VM runs the result."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VM = ROOT / 'bin' / 'nano_vm'
STAGE1 = ROOT / 'bin' / 'nanoc_stage1.nvm'


class SelfhostRegexHostAbi(unittest.TestCase):
    def compile_and_run(self, name):
        with tempfile.TemporaryDirectory() as work:
            module = Path(work) / f'{name}.nvm'
            compiled = subprocess.run(
                [str(VM), str(STAGE1), '--', str(ROOT / 'tests' / f'{name}.nano'), '--emit-nvm', '-o', str(module)],
                capture_output=True, text=True, timeout=300, cwd=ROOT)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            self.assertNotIn('unsupported extern', compiled.stdout + compiled.stderr)
            return subprocess.run([str(VM), str(module)], capture_output=True, text=True, timeout=120)

    def test_split_runs_through_the_declared_regex_imports(self):
        ran = self.compile_and_run('test_split2')
        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
        self.assertEqual(ran.stdout.splitlines(), ['Parts: 3', 'Part 0: a', 'Part 1: b', 'Part 2: c'])

    def test_groups_result_survives_free(self):
        ran = self.compile_and_run('test_std_regex_groups_uaf')
        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)


if __name__ == '__main__':
    unittest.main()
