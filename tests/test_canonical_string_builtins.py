"""I retain exact string results and native trim boundaries."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()


class CanonicalStringBuiltins(unittest.TestCase):
    def command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, expected, (result.stdout + result.stderr)[-5000:])
        return result

    def both(self, source, directory):
        module, c_source, native = (directory / name for name in ('main.nvm', 'main.c', 'native'))
        self.command([COMPILER, source, '--emit-nvm', '-o', module])
        self.command([ROOT / 'bin/nano_vm', '--verify-only', module])
        vm = self.command([ROOT / 'bin/nano_vm', module])
        self.command([ROOT / 'bin/nvm2c', module, '-o', c_source])
        self.command(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', c_source,
                      ROOT / 'bin/nano_aot_runtime.o', '-lm',
                      *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                      '-o', native])
        self.assertEqual(self.command([native]).stdout, vm.stdout)

    def test_existing_string_edges(self):
        with tempfile.TemporaryDirectory(prefix='canonical-string-edges-') as tmp:
            self.both(ROOT / 'tests/nl_functions_string_edges.nano', Path(tmp))

    def test_nested_string_and_boolean_results(self):
        with tempfile.TemporaryDirectory(prefix='canonical-string-results-') as tmp:
            directory = Path(tmp)
            source = directory / 'main.nano'
            source.write_text('''fn starts(value: string) -> bool { return (str_starts_with value "a") }
shadow starts { assert (starts "abc") }
fn ends(value: string) -> bool { return (str_ends_with value "c") }
shadow ends { assert (ends "abc") }
fn join(value: string) -> string { return (str_concat (str_trim value) "bc") }
shadow join { assert (== (join " a ") "abc") }
fn main() -> int {
 let value: string = (str_trim (str_concat " a" "bc "))
 assert (== value "abc")
 assert (starts (join " a "))
 assert (ends value)
 assert (== (str_trim "\\v") "\\v")
 return 0
}
shadow main { assert (== (main) 0) }
''')
            self.both(source, directory)

    def test_wrong_trim_operands_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix='canonical-trim-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory / 'main.nano', directory / 'prior.nvm'
            for expression in ('(str_trim 1)', '(str_trim "a" "b")'):
                with self.subTest(expression=expression):
                    source.write_text('fn main() -> int { ' + expression + ' return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'prior')
                    self.command([COMPILER, source, '--emit-nvm', '-o', output], 1)
                    self.assertEqual(output.read_bytes(), b'prior')


if __name__ == '__main__':
    unittest.main()
