"""I run unchanged global sources through my Nano producer, VM and native C.

This raw producer gate includes every selected source shadow. It does not
replace fresh installed Stage1/Stage2 checking and publication qualification.
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile

from tests.test_owned_global_source import OwnedGlobalSource as _GlobalSources

ROOT = Path(__file__).resolve().parents[1]


class OwnedGlobalProducer(_GlobalSources):
    @classmethod
    def setUpClass(cls):
        retained = os.environ.get('NANO_GLOBAL_PRODUCER_ARTIFACTS')
        cls.temporary = None if retained else tempfile.TemporaryDirectory(prefix='nano-global-producer-')
        cls.work = Path(retained or cls.temporary.name)
        cls.work.mkdir(parents=True, exist_ok=True)
        source = cls.work / 'shadow_driver.nano'
        source.write_text((ROOT / 'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_text())
        cls.shadow_tool = cls.work / 'shadow-driver'
        result = subprocess.run([ROOT / 'bin/nanoc_c', source, '-o', cls.shadow_tool],
                                cwd=ROOT, capture_output=True, text=True, timeout=900)
        (cls.work / 'shadow-driver-build.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def source_route(self, source, accepted, diagnostics=None):
        work = self.work / self._testMethodName
        work.mkdir(exist_ok=True)
        program = work / 'main.nano'
        program.write_text(source)
        # These raw lowering diagnostics are distinct from the C checker diagnostics.
        refusal = {
            'test_immutable_assignment_refused': 'mutable whole global target',
            'test_wrong_global_assignment_type_refused': 'exact declared global value type',
            'test_uninitialized_global_read_in_initializer_refused': 'global initialization',
            'test_same_shape_wrong_union_initializer_refused': 'exact union constructor declaration',
            'test_same_shape_wrong_union_assignment_refused': 'exact union constructor declaration',
            'test_wrong_generic_union_global_alias_refused': 'exact declared global union identity',
            'test_resource_global_remains_refused': 'explicit copyable scalar or union global types',
        }
        sequence = 0

        def run(argv, timeout=60, env=None):
            nonlocal sequence
            sequence += 1
            result = subprocess.run([str(arg) for arg in argv], cwd=ROOT,
                                    capture_output=True, text=True, timeout=timeout, env=env)
            (work / f'{sequence}-command.json').write_text(json.dumps({
                'argv': [str(arg) for arg in argv], 'exit': result.returncode}, indent=2) + '\n')
            (work / f'{sequence}-stdout.log').write_text(result.stdout)
            (work / f'{sequence}-stderr.log').write_text(result.stderr)
            return result

        def success(result):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        for shadows in (False, True):
            with self.subTest(shadows=shadows):
                assembly = work / ('shadows.nasm' if shadows else 'main.nasm')
                module = assembly.with_suffix('.nvm')
                module.write_bytes(b'prior artifact')
                if shadows:
                    result = run([self.shadow_tool, program, 0, 'raw'])
                    if result.returncode == 0:
                        assembly.write_text(result.stdout)
                else:
                    result = run([ROOT / 'bin/nanoisa_emit', program, '-o', assembly])
                if result.returncode == 0:
                    result = run([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                if not accepted:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stdout + result.stderr, refusal[self._testMethodName])
                    self.assertEqual(module.read_bytes(), b'prior artifact')
                    continue
                success(result)
                success(run([ROOT / 'bin/nano_vm', '--verify-only', module]))
                success(run([ROOT / 'bin/nano_vm', module], timeout=10))
                native_source, native = assembly.with_suffix('.c'), assembly.with_suffix('.native')
                success(run([ROOT / 'bin/nvm2c', module, '-o', native_source]))
                compiler = os.environ.get('CC', '/opt/homebrew/opt/llvm/bin/clang'
                    if Path('/opt/homebrew/opt/llvm/bin/clang').is_file() else 'cc')
                success(run([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                             '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                             '-g', native_source, '-o', native]))
                success(run([native], timeout=10, env={**os.environ,
                    'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                    'UBSAN_OPTIONS': 'halt_on_error=1'}))


del _GlobalSources
