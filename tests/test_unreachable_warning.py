"""I keep unreachable warnings nonfatal while checking every statement and shadow."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = '''fn choose() -> int {
    return 7
    return 9
}
shadow choose { assert (== (choose) 7) }
fn main() -> int {
    assert (== (choose) 7)
    return 0
}
'''


class UnreachableWarning(unittest.TestCase):
    def compile(self, compiler, source, output):
        return subprocess.run([ROOT / 'bin' / compiler, source, '-o', output],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_warning_only_program_runs_in_all_native_stages(self):
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / 'source.nano', Path(tmp) / 'program'
                source.write_text(PROGRAM)
                result = self.compile(compiler, source, output)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_type_error_after_return_preserves_output(self):
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / 'source.nano', Path(tmp) / 'program'
                source.write_text(PROGRAM.replace('return 9', 'return "wrong"'))
                output.write_text('previous accepted output')
                result = self.compile(compiler, source, output)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('return', (result.stdout + result.stderr).lower())
                self.assertEqual(output.read_text(), 'previous accepted output')

    def test_warning_does_not_skip_a_failing_shadow(self):
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / 'source.nano', Path(tmp) / 'program'
                source.write_text(PROGRAM.replace('shadow choose { assert (== (choose) 7) }',
                                                 'shadow choose { assert (== (choose) 8) }'))
                output.write_text('previous accepted output')
                result = self.compile(compiler, source, output)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('shadow', (result.stdout + result.stderr).lower())
                self.assertEqual(output.read_text(), 'previous accepted output')


if __name__ == '__main__':
    unittest.main()
