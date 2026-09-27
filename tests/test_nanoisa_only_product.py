"""I publish native and C products through my verified NanoISA translation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()


class NanoisaOnlyProduct(unittest.TestCase):
    def invoke(self, args, env=None, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                env={**os.environ, **(env or {})},
                                capture_output=True, timeout=120)
        self.assertEqual(result.returncode, expected, (result.stdout + result.stderr)[-6000:])
        return result

    def test_native_and_source_products_use_translator(self):
        with tempfile.TemporaryDirectory(prefix='nano product ') as directory:
            work = Path(directory)
            source = work / 'main.nano'
            source.write_text('fn main() -> int { (println "one-ir") return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            calls = work / 'calls'
            wrapper = work / 'translator'
            wrapper.write_text('#!' + sys.executable + '\nimport os,sys\n'
                               f'with open({str(calls)!r}, "a") as log: log.write("translate\\n")\n'
                               f'os.execv({str(ROOT / "bin/nvm2c")!r}, ["nvm2c"] + sys.argv[1:])\n')
            wrapper.chmod(0o755)
            env = {'NANO_NVM2C': str(wrapper)}
            native, c_file = work / 'native', work / 'source.c'
            self.invoke([COMPILER, source, '-o', native], env)
            self.assertEqual(self.invoke([native]).stdout, b'one-ir\n')
            # Source emission has no VM execution dependency.
            self.invoke([COMPILER, source, '--target', 'c', '-o', c_file],
                        {**env, 'NANO_VM': str(work / 'absent-vm')})
            self.assertEqual(calls.read_text(), 'translate\ntranslate\n')
            rebuilt = work / 'rebuilt'
            flags = ['-rdynamic', '-ldl'] if sys.platform.startswith('linux') else []
            self.invoke(['cc', '-std=c11', c_file, ROOT / 'bin/nano_aot_runtime.o',
                         '-lm', *flags, '-o', rebuilt])
            self.assertEqual(self.invoke([rebuilt]).stdout, b'one-ir\n')
            self.assertEqual(list(work.glob('.nano-product.*')), [])

    def test_failed_product_steps_preserve_previous_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-product-failure-') as directory:
            work = Path(directory)
            source, output = work / 'main.nano', work / 'product'
            source.write_text('fn main() -> int { return 0 }\nshadow main { assert true }\n')
            failed = work / 'failed'
            failed.write_text('#!/bin/sh\nexit 1\n')
            failed.chmod(0o755)
            for env in ({'NANO_NVM2C': str(failed)},
                        {'NANO_AOT_RUNTIME': str(work / 'absent-runtime')},
                        {'NANO_CC': str(failed)}, {'NANO_VM': str(failed)}):
                with self.subTest(env=env):
                    output.write_bytes(b'previous product')
                    self.invoke([COMPILER, source, '-o', output], env, expected=1)
                    self.assertEqual(output.read_bytes(), b'previous product')
                    self.assertEqual(list(work.glob('.nano-product.*')), [])


if __name__ == '__main__':
    unittest.main()
