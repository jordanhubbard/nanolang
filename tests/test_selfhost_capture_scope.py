"""I check anonymous bodies in their creation scope in VM and native products."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/selfhost_capture_scope.nano'


class SelfhostCaptureScope(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_lexical_scope_and_refusal_in_both_products(self):
        with tempfile.TemporaryDirectory(prefix='nano-capture-scope-') as tmp:
            work = Path(tmp)
            module, source, binary = [work / name for name in ('scope.nvm', 'scope.c', 'scope')]
            self.checked([ROOT / 'bin/nano_virt', SOURCE, '--emit-nvm', '-o', module])
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            self.assertIn('lexical checker ok\n', vm.stdout)
            self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
            cc = os.environ.get('NANO_NATIVE_TEST_CC') or shutil.which('cc')
            self.assertTrue(cc)
            self.checked([cc, '-std=c11', '-O0', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          source, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary])
            native = self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
            self.assertEqual(native.stdout, vm.stdout)


if __name__ == '__main__':
    unittest.main()
