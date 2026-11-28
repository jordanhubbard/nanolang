"""I preserve a live callable after displaying or reloading its expression."""
import os
from pathlib import Path
import tempfile
import unittest

from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]


class ReplCallableLifetimes(unittest.TestCase):
    def observe(self, lines):
        binary = Path(os.environ.get('NANO_REPL', ROOT / 'bin/nanolang-repl')).resolve()
        self.assertTrue(binary.is_file(), binary)
        with tempfile.TemporaryFile() as source:
            source.write(('\n'.join(lines) + '\n').encode())
            source.seek(0)
            result = run_captured([binary], cwd=ROOT, stdin=source, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertNotIn(b'Undefined', result.stderr)
        self.assertNotIn(b'AddressSanitizer', result.stderr)
        return result.stdout

    def test_display_preserves_binding(self):
        output = self.observe([
            'fn kept(x: int) -> int { return x }',
            'shadow kept { assert (== (kept 7) 7) }',
            'let action: fn(int) -> int = kept',
            'action', 'kept', 'action', '(action 7)', 'action', '(action 8)', ':quit',
        ])
        self.assertIn(b'=> 7', output)
        self.assertIn(b'=> 8', output)

    def test_load_retires_displayed_result(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-callable-') as directory:
            path = Path(directory) / 'loaded.nano'
            path.write_text('fn loaded(x: int) -> int { return (+ x 1) }\n'
                            'shadow loaded { assert (== (loaded 7) 8) }\nloaded\n')
            output = self.observe([f':load {path}', 'loaded', '(loaded 7)', ':quit'])
        self.assertIn(b'=> 8', output)

    def test_reload_retires_ignored_result(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-callable-') as directory:
            path = Path(directory) / 'reloaded.nano'
            path.write_text('fn kept(x: int) -> int { return (+ x 2) }\n'
                            'shadow kept { assert (== (kept 7) 9) }\nkept\n')
            output = self.observe([
                'fn kept(x: int) -> int { return x }',
                'shadow kept { assert (== (kept 7) 7) }',
                'let action: fn(int) -> int = kept',
                f':reload {path}', 'action', '(action 7)', ':quit',
            ])
        self.assertIn(b'=> 9', output)


if __name__ == '__main__':
    unittest.main()
