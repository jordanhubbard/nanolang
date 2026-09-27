"""I retain recursive calls and builtin results under compiler instrumentation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_RECURSION_COMPILER', ROOT / 'bin/nanoc_c'))


class InterpreterRecursiveStack(unittest.TestCase):
    def check(self, indirect=False, fail_shadow=False):
        call = '(walk (- depth 1))'
        binding = ''
        if indirect:
            binding = 'let next: fn(int) -> int = walk'
            call = '(next (- depth 1))'
        source = f'''
fn walk(depth: int) -> int {{
    if (== depth 0) {{ return (str_length "base") }}
    {binding}
    return (+ {call} (str_length "x"))
}}
shadow walk {{ assert (== (walk 128) {133 if fail_shadow else 132}) }}
fn main() -> int {{ return (- (walk 128) 132) }}
shadow main {{ assert (== (main) 0) }}
'''
        with tempfile.TemporaryDirectory(prefix='nano-recursive-stack-') as directory:
            path, output = Path(directory) / 'case.nano', Path(directory) / 'case'
            path.write_text(source)
            output.write_bytes(b'prior artifact')
            result = subprocess.run([COMPILER, path, '-o', output], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
            if fail_shadow:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Shadow test 'walk' FAILED: 1 failure(s)", result.stdout + result.stderr)
                self.assertNotIn('after signal', result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                ran = subprocess.run([output], capture_output=True, text=True, timeout=15)
                self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_recursive_calls_with_builtin_results(self):
        self.check()

    def test_recursive_function_value_with_builtin_results(self):
        self.check(indirect=True)

    def test_failed_recursive_shadow_preserves_output(self):
        self.check(fail_shadow=True)
