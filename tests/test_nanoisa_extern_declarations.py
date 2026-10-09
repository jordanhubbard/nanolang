"""I retain historical extern declarations through verified VM and native products."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExternDeclarations(unittest.TestCase):
    def test_original_declaration_assertions_on_both_frontends_and_installed_stages(self):
        retained = os.environ.get('NANO_EXTERN_DECLARATION_ARTIFACTS')
        temporary = None if retained else tempfile.TemporaryDirectory(prefix='nano-extern-declarations-')
        if temporary:
            self.addCleanup(temporary.cleanup)
        work = Path(retained or temporary.name)
        work.mkdir(parents=True, exist_ok=True)
        source = work / 'driver.nano'
        source.write_text(
            (ROOT / 'tests/nanoisa/fixtures/historical_extern_helpers.nano.txt').read_text()
            + '\n' + (ROOT / 'tests/nanoisa/fixtures/historical_extern_assertions.nano.txt').read_text())
        environment = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                       'UBSAN_OPTIONS': 'halt_on_error=1'}
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        sequence = 0

        def checked(args, expected_stdout=None):
            nonlocal sequence
            result = subprocess.run(list(map(str, args)), cwd=ROOT, env=environment,
                                    capture_output=True, text=True, timeout=180)
            (work / f'{sequence:02d}.log').write_text(result.stdout + result.stderr)
            sequence += 1
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if expected_stdout is not None:
                self.assertEqual(result.stdout, expected_stdout)

        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler):
                module = work / (compiler + '.nvm')
                c_source = work / (compiler + '.c')
                native = work / (compiler + '.native')
                checked([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', module])
                checked([ROOT / 'bin/nano_vm', '--verify-only', module])
                checked([ROOT / 'bin/nano_vm', module], 'I passed extern declaration selection.\n')
                checked([ROOT / 'bin/nvm2c', module, '-o', c_source])
                checked([*cc, '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=address,undefined', c_source, '-lm', '-ldl', '-o', native])
                checked([native], 'I passed extern declaration selection.\n')


if __name__ == '__main__':
    unittest.main()
