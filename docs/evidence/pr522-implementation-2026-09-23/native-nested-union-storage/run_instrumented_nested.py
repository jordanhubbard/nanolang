"""I run unchanged nested fixtures through a fresh instrumented translator."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CLANG = '/opt/homebrew/opt/llvm/bin/clang'
FLAGS = '-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'


def main():
    with tempfile.TemporaryDirectory(prefix='nano-nested-instrumented-') as directory:
        work = Path(directory)
        command = ['make', '-j1', f'CC={CLANG} {FLAGS}', f'OBJ_DIR={work / "obj"}',
                   f'BIN_DIR={work / "bin"}',
                   f'FILE_PUBLIC_LIBRARY={work / "lib/libnano_file_runtime.a"}', 'nvm2c']
        subprocess.run(command, cwd=ROOT, check=True)
        for name in ('nvm2c', 'nvm2c_shape'):
            symbols = subprocess.check_output(['nm', '-u', work / f'obj/nanoisa/{name}.o'], text=True)
            if '__asan_' not in symbols or '__ubsan_' not in symbols:
                raise RuntimeError(f'I require actual instrumentation in {name}.o')
        translator = work / 'bin/nvm2c'
        environment = {'NANO_NVM2C': str(translator), 'NANO_NATIVE_TEST_CC': CLANG,
                       'NANO_CC': CLANG, 'NANO_CFLAGS': '-O1 -g ' + FLAGS,
                       'NANO_LDFLAGS': '-fsanitize=address,undefined',
                       'ASAN_OPTIONS': 'detect_leaks=1:detect_stack_use_after_return=1:halt_on_error=1',
                       'UBSAN_OPTIONS': 'halt_on_error=1'}
        original = subprocess.run
        routed_count = 0

        def routed(args, *positional, **kwargs):
            nonlocal routed_count
            if isinstance(args, (list, tuple)) and args and Path(args[0]) == ROOT / 'bin/nvm2c':
                args = [str(translator), *args[1:]]
                # Raw helpers set their own leak/halt options. I preserve them
                # and add UAR for the instrumented translator invocation.
                env = dict(kwargs.get('env') or os.environ)
                env['ASAN_OPTIONS'] = env.get('ASAN_OPTIONS', '') + ':detect_stack_use_after_return=1'
                kwargs['env'] = env
                routed_count += 1
            return original(args, *positional, **kwargs)

        with patch.dict(os.environ, environment), patch('subprocess.run', routed):
            suite = unittest.defaultTestLoader.loadTestsFromNames([
                'tests.test_native_indexed_union_payloads',
                'tests.test_canonical_aggregate_formatting'])
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(f'I routed {routed_count} raw translator invocations through verified ASan/UBSan objects.')
        return 0 if result.wasSuccessful() and routed_count else 1


if __name__ == '__main__':
    raise SystemExit(main())
