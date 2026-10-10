"""I run the existing selected-owner acceptance sources through NanoVirt/VM/C."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
from tests.test_selected_variant_ownership import SelectedVariantOwnership as _Selected
from tests.test_generic_selected_ownership import GenericSelectedOwnership as _Generic

ROOT = Path(__file__).resolve().parents[1]

class _SourceRoute:
    def program(self, source, accepted, diagnostics=None):
        self.source_route(source, accepted, diagnostics)

    def source_route(self, source, accepted, diagnostics=None):
        retained = os.environ.get('NANO_SELECTED_SOURCE_ARTIFACTS')
        temporary = None if retained else tempfile.TemporaryDirectory(prefix='nano-owned-union-source-')
        work = Path(retained) / self.__class__.__name__ / self._testMethodName if retained else Path(temporary.name)
        work.mkdir(parents=True, exist_ok=True)
        try:
            program, module = work / 'main.nano', work / 'main.nvm'
            program.write_text(source)
            module.write_bytes(b'prior artifact')
            sequence = 0
            def run(command, timeout=120, env=None):
                nonlocal sequence
                sequence += 1
                result = subprocess.run([str(arg) for arg in command], cwd=ROOT,
                    capture_output=True, text=True, timeout=timeout, env=env)
                (work / f'{sequence}-command.json').write_text(json.dumps({
                    'argv': [str(arg) for arg in command], 'exit': result.returncode}, indent=2) + '\n')
                (work / f'{sequence}-stdout.log').write_text(result.stdout)
                (work / f'{sequence}-stderr.log').write_text(result.stderr)
                return result
            compiler = ROOT / 'bin' / getattr(self, 'source_compiler', 'nano_virt')
            result = run([compiler, program, '--emit-nvm', '-o', module])
            messages = result.stdout + result.stderr
            if not accepted:
                self.assertGreater(result.returncode, 0, messages)
                self.assertRegex(messages, diagnostics['nanoc_c'] if diagnostics else '(?i)(ownership|resource|moved)')
                self.assertEqual(module.read_bytes(), b'prior artifact')
                return
            self.assertEqual(result.returncode, 0, messages)
            for command in ([ROOT / 'bin/nano_vm', '--verify-only', module],
                            [ROOT / 'bin/nano_vm', module]):
                result = run(command, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            native_source, native = work / 'main.c', work / 'native'
            result = run([ROOT / 'bin/nvm2c', module, '-o', native_source])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            compiler = os.environ.get('CC', '/opt/homebrew/opt/llvm/bin/clang' if Path('/opt/homebrew/opt/llvm/bin/clang').is_file() else 'cc')
            result = run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                          '-g', native_source, '-o', native], timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = run([native], timeout=10, env={**os.environ,
                'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            if temporary:
                temporary.cleanup()

class SelectedUnionCSource(_SourceRoute, _Selected):
    pass

class GenericSelectedUnionCSource(_SourceRoute, _Generic):
    def check(self, source, accepted):
        self.source_route(source, accepted)

del _Selected, _Generic
