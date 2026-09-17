"""I preserve ordinary lexical underscore bindings in both native producers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = Path(os.environ.get('NANOLANG_DISCARD_COMPILER_ROOT', ROOT/'bin'))
SOURCE = ROOT/'tests/nanoisa/fixtures/native_underscore_bindings.nano'


class NativeUnderscoreBindings(unittest.TestCase):
    def run_checked(self, args, timeout=120):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_native_compiler_stages(self):
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-underscore-') as tmp:
                output = Path(tmp)/'program'
                self.run_checked([COMPILERS/compiler, SOURCE, '-o', output])
                self.run_checked([output], timeout=10)

    def test_interpreter_and_vm(self):
        self.run_checked([ROOT/'bin/nano', SOURCE])
        with tempfile.TemporaryDirectory(prefix='nano-underscore-vm-') as tmp:
            module = Path(tmp)/'program.nvm'
            self.run_checked([ROOT/'bin/nano_virt', SOURCE, '-o', module])
            self.run_checked([ROOT/'bin/nano_vm', '--verify-only', module])
            self.run_checked([ROOT/'bin/nano_vm', module])

    def test_cseed_cleanup_uses_each_emitted_binding(self):
        import re
        with tempfile.TemporaryDirectory(prefix='nano-underscore-cleanup-') as tmp:
            output = Path(tmp)/'program'
            self.run_checked([COMPILERS/'nanoc_c', SOURCE, '--keep-c', '-o', output])
            code = output.with_suffix('.c').read_text()
            names = re.findall(r'gc_release\((__nl_underscore_\d+)\)', code)
            self.assertGreaterEqual(len(set(names)), 2)
            self.assertNotIn('gc_release(_);', code)
            self.assertIn(str(SOURCE), code)

    def test_legacy_c_par_let_alias(self):
        source = """fn legacy() -> int {
 let _: int = 1
 par-let _=7 other=8 in (+ _ other)
 return _
}
shadow legacy { assert (== (legacy) 7) }
fn main() -> int { assert (== (legacy) 7) return 0 }
"""
        with tempfile.TemporaryDirectory(prefix='nano-underscore-par-let-') as tmp:
            path, output = Path(tmp)/'input.nano', Path(tmp)/'program'
            path.write_text(source)
            self.run_checked([ROOT/'bin/nano', path])
            self.run_checked([COMPILERS/'nanoc_c', path, '-o', output])
            self.run_checked([output])

    def test_failed_shadow_preserves_output(self):
        source = SOURCE.read_text().replace('assert (== order 123)', 'assert (== order 124)')
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-underscore-shadow-') as tmp:
                path, output = Path(tmp)/'input.nano', Path(tmp)/'program'
                path.write_text(source)
                output.write_text('previous accepted output')
                result = subprocess.run([COMPILERS/compiler, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('shadow', (result.stdout + result.stderr).lower())
                self.assertEqual(output.read_text(), 'previous accepted output')


if __name__ == '__main__':
    unittest.main()
