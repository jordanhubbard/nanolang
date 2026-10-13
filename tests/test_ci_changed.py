"""I check that the CI change classifier skips only pure documentation pushes."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ci_changed  # noqa: E402


class CiChanged(unittest.TestCase):
    def test_documentation_only_is_not_code(self):
        for paths in (['docs/ROADMAP.md'], ['docs/evidence/x/README.md', 'CHANGELOG.md'], ['README.md'],
                      ['docs/RELEASE_5.1_STATUS.md', 'docs/NANOISA_ONLY.md', 'CHANGELOG.md']):
            with self.subTest(paths=paths):
                self.assertFalse(ci_changed.classify(paths))

    def test_any_code_or_build_file_is_code(self):
        for paths in (['src/typechecker.c'], ['docs/ROADMAP.md', 'src/typechecker.c'], ['Makefile.gnu'],
                      ['.github/workflows/ci.yml'], ['tests/test_x.py'], ['scripts/fast_gate.py'],
                      ['docs/ROADMAP.md', 'Makefile.gnu'], ['src_nano/nanoc_v06.nano'], ['docs/tool.py']):
            with self.subTest(paths=paths):
                self.assertTrue(ci_changed.classify(paths))

    def test_unknown_or_empty_input_runs_the_long_jobs(self):
        for paths in ([], [''], ['   ']):
            with self.subTest(paths=paths):
                self.assertTrue(ci_changed.classify(paths))

    def test_command_output(self):
        for text, expected in (('docs/a.md\n', 'code=false'), ('docs/a.md\nsrc/a.c\n', 'code=true'), ('', 'code=true')):
            with self.subTest(text=text):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/ci_changed.py')], input=text,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.stdout.strip(), expected)


if __name__ == '__main__':
    unittest.main()
