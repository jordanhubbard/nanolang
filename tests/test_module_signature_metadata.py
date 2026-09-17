#!/usr/bin/env python3
"""I retain complete callback annotations across compiled module boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOC', ROOT / 'bin/nanoc_c')).resolve()

MODULE = '''fn echo(values: array<array<int>>) -> array<array<int>> { return values }
shadow echo { let values: array<array<int>> = [[7]] let result: array<array<int>> = (echo values) assert (== (at (at result 0) 0) 7) }
pub fn apply(f: fn(array<array<int>>) -> array<array<int>>, value: array<array<int>>) -> array<array<int>> { return (f value) }
shadow apply { let values: array<array<int>> = [[7]] let result: array<array<int>> = (apply echo values) assert (== (at (at result 0) 0) 7) }
'''
SOURCE = '''from "callbacks.nano" import apply
fn identity(value: array<array<int>>) -> array<array<int>> { return value }
shadow identity { let values: array<array<int>> = [[7]] let result: array<array<int>> = (identity values) assert (== (at (at result 0) 0) 7) }
fn main() -> int { let values: array<array<int>> = [[7]] let result: array<array<int>> = (apply identity values) assert (== (at (at result 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
'''


class ModuleSignatureMetadata(unittest.TestCase):
    def check(self, mismatched):
        with tempfile.TemporaryDirectory(prefix='nano-imported-signatures-') as name:
            directory = Path(name)
            (directory / 'callbacks.nano').write_text(MODULE)
            source = SOURCE
            if mismatched:
                source = source.replace('fn identity(value: array<array<int>>) -> array<array<int>>',
                                        'fn identity(value: array<array<string>>) -> array<array<string>>')
                source = source.replace('shadow identity { let values: array<array<int>> = [[7]] let result: array<array<int>> = (identity values) assert (== (at (at result 0) 0) 7) }',
                                        'shadow identity { let values: array<array<string>> = [["x"]] let result: array<array<string>> = (identity values) assert (== (at (at result 0) 0) "x") }')
            path, output = directory / 'main.nano', directory / 'program'
            path.write_text(source)
            output.write_bytes(b'previous artifact')
            result = subprocess.run([COMPILER, path, '-o', output], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
            diagnostic = result.stdout + result.stderr
            if mismatched:
                self.assertNotEqual(result.returncode, 0, diagnostic)
                self.assertEqual(output.read_bytes(), b'previous artifact')
            else:
                self.assertEqual(result.returncode, 0, diagnostic)
                run = subprocess.run([output], capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_imported_nested_array_callback_executes(self):
        self.check(False)

    def test_imported_nested_array_callback_mismatch_preserves_output(self):
        self.check(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
