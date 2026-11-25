"""I retain source binding identity when native C requires another spelling."""
from pathlib import Path
import os
import re
import tempfile
import unittest
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get('NANOLANG_KEYWORD_TOOL_ROOT', ROOT/'bin'))
SOURCE = ROOT/'tests/nanoisa/fixtures/native_keyword_bindings.nano'


class NativeKeywordBindings(unittest.TestCase):
    def checked(self, argv, *, build=False):
        result = run_captured(argv, cwd=ROOT, timeout=60, build=build)
        self.assertEqual(result.returncode, 0,
                         result.stdout.decode(errors='replace') +
                         result.stderr.decode(errors='replace'))
        return result

    def test_source_vm_and_cseed_values(self):
        self.checked([TOOLS/'nano', SOURCE])
        with tempfile.TemporaryDirectory(prefix='nano-keywords-') as temporary:
            module, binary = Path(temporary)/'program.nvm', Path(temporary)/'program'
            self.checked([TOOLS/'nano_virt', SOURCE, '-o', module], build=True)
            self.checked([TOOLS/'nano_vm', '--verify-only', module])
            self.assertEqual(self.checked([TOOLS/'nano_vm', module]).stdout,
                             b'C-KEYWORD-OK\n')
            self.checked([TOOLS/'nanoc_c', SOURCE, '--keep-c', '-o', binary], build=True)
            self.assertEqual(self.checked([binary]).stdout, b'C-KEYWORD-OK\n')
            code = binary.with_suffix('.c').read_text()
            releases = re.findall(r'gc_release\((__nl_keyword_l_\d+)\)', code)
            self.assertGreaterEqual(len(set(releases)), 2)
            self.assertNotIn('gc_release(signed);', code)
            self.assertIn('__nl_keyword_p_7369676e6564', code)

    def test_keyword_handler_parameters_and_captures(self):
        source = """effect Ask { ask : int -> int }
fn exercise() -> int {
 let mut signed: int = 3
 let first = handle { perform Ask.ask(7) } with {
  ask register -> { set signed (+ signed register) signed }
 }
 let second = handle { perform Ask.ask(2) } with {
  ask register -> { let register: int = (+ register 1) (+ register signed) }
 }
 return (+ first second)
}
shadow exercise { assert (== (exercise) 23) }
fn main() -> int { assert (== (exercise) 23) return 0 }
shadow main { assert (== (main) 0) }
"""
        with tempfile.TemporaryDirectory(prefix='nano-keyword-effects-') as temporary:
            path, binary = Path(temporary)/'input.nano', Path(temporary)/'program'
            path.write_text(source)
            self.checked([TOOLS/'nanoc_c', path, '-o', binary], build=True)
            self.checked([binary])

    def test_failed_shadow_keeps_prior_output(self):
        source = SOURCE.read_text().replace(
            'shadow keyword_scalar { assert (== (keyword_scalar 3) 13) }',
            'shadow keyword_scalar { assert (== (keyword_scalar 3) 14) }')
        self.assertNotEqual(source, SOURCE.read_text())
        with tempfile.TemporaryDirectory(prefix='nano-keyword-failure-') as temporary:
            path, output = Path(temporary)/'input.nano', Path(temporary)/'program'
            path.write_text(source)
            output.write_bytes(b'previous accepted output\x00\xff')
            result = run_captured([TOOLS/'nanoc_c', path, '-o', output],
                                  cwd=ROOT, timeout=60, build=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'shadow', result.stdout.lower() + result.stderr.lower())
            self.assertEqual(output.read_bytes(), b'previous accepted output\x00\xff')


if __name__ == '__main__':
    unittest.main()
