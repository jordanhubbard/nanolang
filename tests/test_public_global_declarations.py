"""I retain public-global declaration metadata before import binding."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicGlobalDeclarations(unittest.TestCase):
    def test_parser_visibility_survives_storage_and_type_backfill(self):
        with tempfile.TemporaryDirectory(prefix='nano-public-global-parser-') as directory:
            work = Path(directory)
            module, source, binary = (work / name for name in ('parser.nvm', 'parser.c', 'parser'))
            env = {**os.environ, 'NANO_BUILD_CACHE': str(work / 'cache'),
                   'NANO_SHADOW_TIMEOUT_SECONDS': '60', 'ASAN_OPTIONS': 'detect_leaks=1'}
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc')))
            commands = [
                [ROOT / 'bin/nano_virt', ROOT / 'tests/fixtures/public_global_parser.nano', '--emit-nvm', '-o', module],
                [ROOT / 'bin/nano_vm', '--verify-only', module],
                [ROOT / 'bin/nano_vm', module],
                [ROOT / 'bin/nvm2c', module, '-o', source],
                cc + ['-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all', source, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary],
                [binary],
            ]
            for command in commands:
                result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
