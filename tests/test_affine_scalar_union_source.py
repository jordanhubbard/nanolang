"""I retain distinct concrete scalar-union instances in my affine source path."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/fixtures/affine_scalar_union_instances.nano'


class AffineScalarUnionSource(unittest.TestCase):
    def command(self, *args, env=None):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT,
                                capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_distinct_instances_verify_and_execute_in_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-union-source-') as raw:
            work = Path(raw)
            assembly = self.command(ROOT / 'bin/nanoisa_emit', FIXTURE).stdout
            self.assertIn('.types 1 0 2', assembly)
            self.assertIn('AGG_PACK 1 0 0 1', assembly)
            self.assertIn('AGG_PACK 1 1 1 1', assembly)
            self.assertIn('.parameters 2 union', assembly)
            self.assertIn('.parameters 3 union', assembly)
            nasm = work / 'instances.nasm'
            module = work / 'instances.nvm'
            native_source = work / 'instances.c'
            native = work / 'instances'
            nasm.write_text(assembly)
            self.command(ROOT / 'bin/nanoisa', 'asm', nasm, '-o', module)
            self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
            self.command(ROOT / 'bin/nano_vm', module)
            dumped = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
            self.assertIn('.types 1 0 2', dumped)
            self.assertIn('Choice<int,string>', dumped)
            self.assertIn('Choice<float,bool>', dumped)
            self.command(ROOT / 'bin/nvm2c', module, '-o', native_source)
            compiler = os.environ.get('NANO_NATIVE_TEST_CC', '/opt/homebrew/opt/llvm/bin/clang')
            self.command(compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                         native_source, '-o', native)
            runtime = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'}
            self.command(native, env=runtime)


if __name__ == '__main__':
    unittest.main()
