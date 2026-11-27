"""I reject reserved declarations normally and preserve native publication boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_REJECTED_DECLARATION_COMPILER', ROOT/'bin/nanoc_c'))


class RejectedDeclarations(unittest.TestCase):
    def invoke(self, directory, source, output):
        path = directory/'source.nano'
        path.write_text(source)
        return subprocess.run([str(COMPILER), str(path), '-o', str(output)], cwd=ROOT,
                              env=dict(os.environ, TMPDIR=str(directory)),
                              capture_output=True, text=True, timeout=30)

    def test_reserved_declaration_with_and_without_shadow(self):
        for shadow in [False, True]:
            for prior in [False, True]:
                with self.subTest(shadow=shadow, prior=prior), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    output = directory/'program'
                    if prior:
                        output.write_bytes(b'prior-native-output')
                    source = 'fn bstr_utf8_char_at(x: int, y: int) -> int { return (+ x y) }\n'
                    if shadow:
                        source += 'shadow bstr_utf8_char_at { assert (== (bstr_utf8_char_at 1 2) 3) }\n'
                    source += 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
                    result = self.invoke(directory, source, output)
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn("Cannot redefine built-in function 'bstr_utf8_char_at'", result.stderr)
                    if shadow:
                        self.assertIn("Shadow test for undefined function 'bstr_utf8_char_at'", result.stderr)
                    if prior:
                        self.assertEqual(output.read_bytes(), b'prior-native-output')
                    else:
                        self.assertFalse(output.exists())

    def test_legal_renamed_function_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory/'program'
            source = ('fn bstr_custom(x: int, y: int) -> int { return (+ x y) }\n'
                      'shadow bstr_custom { assert (== (bstr_custom 1 2) 3) }\n'
                      'fn main() -> int { assert (== (bstr_custom 6 8) 14) return 0 }\n'
                      'shadow main { assert (== (main) 0) }\n')
            result = self.invoke(directory, source, output)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stdout+executed.stderr)


if __name__ == '__main__':
    unittest.main()
