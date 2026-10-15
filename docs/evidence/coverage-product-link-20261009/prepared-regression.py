"""I preserve linker flag fallback and override in self-hosted native products."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get('NANO_TEST_ROOT', Path(__file__).resolve().parents[1]))


class NativeProductLinkFlags(unittest.TestCase):
    def test_instrumented_object_uses_effective_link_flags(self):
        override = os.environ.get('NANO_PRODUCT_LINK_COMPILER')
        compilers = [shlex.split(override)] if override else [
            [str(ROOT/'bin'/stage)] for stage in ('nanoc_stage1', 'nanoc_stage2')]
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        configurations = [
            ('fallback', {'LDFLAGS': '--coverage'}, True),
            ('empty_override', {'LDFLAGS': '--coverage', 'NANO_LDFLAGS': ''}, True),
            ('override', {'LDFLAGS': '-lmissing_outer_link_flag', 'NANO_LDFLAGS': '--coverage'}, True),
            ('invalid_override', {'LDFLAGS': '--coverage', 'NANO_LDFLAGS': '-lmissing_explicit_link_flag'}, False),
            ('missing', {}, False),
        ]
        for compiler in compilers:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-product-link-flags-') as tmp:
                work = Path(tmp)
                (work/'bin').mkdir()
                runtime = work/'runtime.c'
                runtime.write_text('int covered_answer(void) { return 42; }\n')
                built = subprocess.run([*cc, '--coverage', '-c', str(runtime), '-o', str(work/'bin/nano_aot_runtime.o')],
                                       cwd=work, capture_output=True, text=True, timeout=30)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                source = work/'hello.nano'
                source.write_text('fn main()->int{return 0}\nshadow main {assert (== (main) 0)}\n')
                for name, flags, succeeds in configurations:
                    with self.subTest(configuration=name):
                        env = dict(os.environ)
                        for key in ('NANO_LDFLAGS', 'LDFLAGS', 'NANO_CFLAGS'):
                            env.pop(key, None)
                        env.update(flags)
                        env.update(NANOLANG_ROOT=str(work), NANO_NVM2C=str(ROOT/'bin/nvm2c'),
                                   NANO_VM=str(ROOT/'bin/nano_vm'), CC=shlex.join(cc), NANO_CC=shlex.join(cc))
                        output = work/name
                        output.write_bytes(b'prior-output')
                        result = subprocess.run(compiler+[str(source), '-o', str(output)], cwd=ROOT, env=env,
                                                capture_output=True, text=True, timeout=120)
                        if succeeds:
                            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                            executed = subprocess.run([str(output)], cwd=work, env=env, capture_output=True, text=True, timeout=30)
                            self.assertEqual(executed.returncode, 0, executed.stdout+executed.stderr)
                        else:
                            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                            self.assertEqual(output.read_bytes(), b'prior-output')


if __name__ == '__main__':
    unittest.main(verbosity=2)
