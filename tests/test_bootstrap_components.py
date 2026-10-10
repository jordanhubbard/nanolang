"""I exercise component and diagnostic-profile Make recipes without legacy emission."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUB = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['COMPONENT_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
out = pathlib.Path(args[args.index('-o') + 1])
if os.environ.get('FAIL_COMPILE') == out.name:
    raise SystemExit(23)
out.parent.mkdir(exist_ok=True)
status = 19 if os.environ.get('FAIL_ENTRY') == out.name else 0
out.write_text('#!/bin/sh\\nexit ' + str(status) + '\\n')
out.chmod(0o755)
'''


class BootstrapComponents(unittest.TestCase):
    def run_recipe(self, profile=False, **failures):
        with tempfile.TemporaryDirectory(prefix='nano-component-recipes-') as tmp:
            root = Path(tmp)
            shutil.copyfile(ROOT / 'Makefile.gnu', root / 'Makefile.gnu')
            compiler = root / 'compiler'
            compiler.write_text(STUB)
            compiler.chmod(0o755)
            (root / 'src_nano').mkdir()
            (root / 'bin').mkdir()
            for driver in ('parser', 'typecheck', 'nanoisa'):
                (root / 'src_nano' / (driver + '_driver.nano')).touch()
            calls = root / 'calls.jsonl'
            args = [os.environ.get('MAKE_BIN', 'make'), '-s', '-f', 'Makefile.gnu',
                    '-o', 'build' if profile else '.stage1.built',
                    'bootstrap-profile' if profile else '.stage3.built',
                    'COMPILER=' + ('/usr/bin/false' if profile else str(compiler)),
                    'COMPILER_C=' + str(compiler), 'TIMEOUT_CMD=', 'UNAME_S=Linux',
                    'BOOTSTRAP_TMPDIR=' + str(root)]
            result = subprocess.run(args, cwd=root, capture_output=True, text=True,
                                    env={**os.environ, 'COMPONENT_CALLS': str(calls), **failures}, timeout=30)
            return result, [json.loads(line) for line in calls.read_text().splitlines()], (
                (root / '.stage2.built').exists(), (root / '.stage3.built').exists())

    def test_nested_make_exports_flags_needed_by_instrumented_native_link(self):
        with tempfile.TemporaryDirectory(prefix='nano-make-link-flags-') as tmp:
            root = Path(tmp)
            shutil.copyfile(ROOT / 'Makefile.gnu', root / 'Makefile.gnu')
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or
                            os.environ.get('CC') or 'cc')
            (root / 'main.c').write_text('int main(void) { return 0; }\n')
            (root / 'probe.py').write_text(
                'import os, shlex, subprocess\n'
                f'cc = {cc!r}\n'
                'subprocess.run(cc + ["--coverage", "-c", "main.c", "-o", "main.o"], check=True)\n'
                'subprocess.run(cc + ["main.o", "-o", "native"] + '
                'shlex.split(os.environ.get("LDFLAGS", "")), check=True)\n'
                'subprocess.run(["./native"], check=True)\n')
            (root / 'probe.mk').write_text(
                'probe-leaf:\n\t@python3 probe.py\n'
                'probe-nested:\n\t@$(MAKE) --no-print-directory -f Makefile.gnu -f probe.mk probe-leaf\n')
            result = subprocess.run([os.environ.get('MAKE_BIN', 'make'), '-s',
                                     '-f', 'Makefile.gnu', '-f', 'probe.mk',
                                     'probe-nested', 'LDFLAGS=--coverage'], cwd=root,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_emitter_driver_executes_in_vm_and_sanitized_native(self):
        with tempfile.TemporaryDirectory(prefix='nano-emitter-component-') as tmp:
            root = Path(tmp)
            module, source, obj, binary = [root / name for name in
                                           ('emitter.nvm', 'emitter.c', 'emitter.o', 'emitter')]
            compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc')))
            commands = [
                [ROOT / 'bin/nano_virt', ROOT / 'src_nano/nanoisa_driver.nano', '--emit-nvm', '-o', module],
                [ROOT / 'bin/nvm2c', module, '-o', source],
                [*compiler, '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                 '-c', source, '-o', obj],
                [*compiler, '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                 obj, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                 *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                 *shlex.split(os.environ.get('LDFLAGS', '')), '-o', binary],
            ]
            for command in commands:
                result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                        text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for command in ([ROOT / 'bin/nano_vm', module], [binary]):
                result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                        text=True, timeout=180,
                                        env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout, 'I passed my NanoISA emitter entry assertions.\n')

    def test_default_components_build_and_execute_nanoisa(self):
        result, calls, stamps = self.run_recipe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([Path(args[0]).name for args in calls],
                         ['parser_driver.nano', 'typecheck_driver.nano', 'nanoisa_driver.nano'])
        self.assertEqual(stamps, (True, True))
        self.assertIn('nanoisa_emitter entry assertions passed', result.stdout)

    def test_failed_component_compile_or_entry_refuses_validation_stamp(self):
        for failure in ('FAIL_COMPILE', 'FAIL_ENTRY'):
            with self.subTest(failure=failure):
                result, calls, stamps = self.run_recipe(**{failure: 'nanoisa_emitter'})
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(stamps[1])
                self.assertEqual(Path(calls[-1][0]).name, 'nanoisa_driver.nano')

    def test_profile_uses_diagnostic_seed_and_propagates_failures(self):
        for failure in (None, 'FAIL_COMPILE', 'FAIL_ENTRY'):
            with self.subTest(failure=failure):
                result, calls, stamps = self.run_recipe(True, **({failure: 'nanoisa_emitter_p'} if failure else {}))
                self.assertEqual([Path(args[0]).name for args in calls],
                                 ['parser_driver.nano', 'typecheck_driver.nano', 'nanoisa_driver.nano'])
                self.assertTrue(all('-pg' in args for args in calls))
                if failure:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('Bootstrap Profile Complete', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
