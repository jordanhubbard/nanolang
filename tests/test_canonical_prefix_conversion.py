"""I retain prefix conversion through canonical bytecode and native output."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CanonicalPrefixConversion(unittest.TestCase):
    def command(self, args, expected=0):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def exercise(self, source, work, compilers=('nano_virt', 'nanoc_stage1', 'nanoc_stage2')):
        for compiler in compilers:
            with self.subTest(compiler=compiler):
                module, c_source, native = (work / name for name in ('main.nvm', 'main.c', 'native'))
                self.command([ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module])
                self.command([ROOT/'bin/nano_vm', '--verify-only', module])
                self.command([ROOT/'bin/nano_vm', module])
                self.command([ROOT/'bin/nvm2c', module, '-o', c_source])
                self.command([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              c_source, '-lm', '-o', native])
                self.command([native])

    def test_retained_prefix_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            self.exercise(ROOT/'tests/unit/test_legacy_binary64_parse.nano', Path(directory))

    def test_result_context_and_single_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work/'source.nano'
            source.write_text('''let mut calls: int = 0
fn text() -> string { set calls (+ calls 1) return "2.5suffix" }
shadow text { assert true }
fn parse(value: string) -> float { return (string_to_float value) }
shadow parse { assert (== (parse "-1.25tail") -1.25) }
fn main() -> int {
 let value = (string_to_float (text))
 assert (== value 2.5)
 assert (== calls 1)
 assert (== (+ (string_to_float "1.5") (parse "2.0")) 3.5)
 assert (== (float_to_string (parse "-0suffix")) "-0.0")
 return 0
}
shadow main { assert (== (main) 0) }
''')
            self.exercise(source, work)

    def test_wrong_operands_preserve_output(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, output = work/'source.nano', work/'prior.nvm'
            for expression in ('(string_to_float 1)', '(string_to_float true)',
                               '(string_to_float "1" "2")'):
                source.write_text('fn main() -> int { ' + expression +
                                  ' return 0 }\nshadow main { assert true }\n')
                for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(expression=expression, compiler=compiler):
                        output.write_bytes(b'prior')
                        result = self.command([ROOT/'bin'/compiler, source, '--emit-nvm', '-o', output], 1)
                        self.assertEqual(output.read_bytes(), b'prior')
                        self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse error')

    def test_qualified_declaration_keeps_its_result_type(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            dependency = work/'helpers.nano'
            dependency.write_text('module helpers\npub fn string_to_float(value: string) -> int { return 7 }\n'
                                  'shadow string_to_float { assert (== (string_to_float "x") 7) }\n')
            source = work/'source.nano'
            source.write_text('module "' + str(dependency) + '" as helpers\n'
                              'fn main() -> int { let value = (helpers.string_to_float "x") '
                              'assert (== value 7) return 0 }\nshadow main { assert (== (main) 0) }\n')
            # My C seed reserves builtin declaration names; selfhost module
            # binding keeps the existing qualified declaration identity.
            result = self.command([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', work/'reserved.nvm'], 1)
            self.assertIn("already defined", result.stderr)
            self.exercise(source, work, ('nanoc_stage1', 'nanoc_stage2'))

    def test_declared_local_function_keeps_its_result_type(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work/'source.nano'
            source.write_text('fn string_to_float(value: string) -> int { return 7 }\n'
                              'shadow string_to_float { assert true }\n'
                              'fn main() -> int { let value = (string_to_float "x") '
                              'assert (== value 7) return 0 }\nshadow main { assert (== (main) 0) }\n')
            self.exercise(source, work, ('nanoc_stage1', 'nanoc_stage2'))


if __name__ == '__main__':
    unittest.main()
