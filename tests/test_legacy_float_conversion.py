"""I check the repaired interpreter and both legacy C emitter paths."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''extern fn get_argc() -> int
extern fn get_argv(index: int) -> string
fn convert(value: float) -> int { return (cast_int value) }
shadow convert { assert (== (convert 2.5) 2) assert (== (convert -2.5) -2) }
fn main() -> int {
 let text: string = (get_argv (- (get_argc) 1))
 (println (convert (string_to_float text)))
 return 0
}
shadow main { assert true }
'''

class LegacyFloatConversion(unittest.TestCase):
    def test_interpreter_and_compiler_stages_share_checked_float_policy(self):
        with tempfile.TemporaryDirectory(prefix='nano-legacy-cast-') as tmp:
            work = Path(tmp)
            source = work / 'convert.nano'
            source.write_text(SOURCE)
            commands = [('interpreter', [ROOT/'bin/nano', source])]
            for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                binary = work/name
                result = subprocess.run([ROOT/'bin'/name, source, '-o', binary],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                    env={**os.environ, 'NANO_CFLAGS': '-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all'})
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                commands.append((name, [binary]))
            cases = [('2.5', '2'), ('-2.5', '-2'), ('0', '0'), ('-0', '0'),
                     ('0.999', '0'), ('-0.999', '0'),
                     ('', '0'), ('not a number', '0'), (' -12.5 trailing', '-12'),
                     ('-9223372036854775808', '-9223372036854775808'),
                     ('9223372036854774784', '9223372036854774784')]
            for name, command in commands:
                for value, expected in cases:
                    with self.subTest(backend=name, value=value):
                        result = subprocess.run([*command, value], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(result.stdout, expected+'\n')
                        self.assertNotIn('runtime error:', result.stderr)
                for value in ('nan', 'inf', '-inf', '9223372036854775808', '-9223372036854777856'):
                    with self.subTest(backend=name, value=value):
                        result = subprocess.run([*command, value], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('I cannot convert this float to int', result.stderr)
                        self.assertNotIn('runtime error:', result.stderr)

if __name__ == '__main__':
    unittest.main()
