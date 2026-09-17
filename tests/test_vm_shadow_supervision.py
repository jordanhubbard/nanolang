"""I require verified VM shadows to return through their completion channel."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VM = ROOT / 'bin/nano_vm'

class VMShadowSupervision(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nano-vm-shadows-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.environment = dict(os.environ)
        self.environment.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)

    def module(self, body, declarations=''):
        source, output = self.directory/'input.nasm', self.directory/'input.nvm'
        source.write_text(declarations + '.entry main\n.function main 0 0 0 int 1\n' + body + '\n.end\n')
        result = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', output],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output

    def run_vm(self, path, *options, budget=None, shadows=True):
        environment = dict(self.environment)
        if budget is not None: environment['NANO_SHADOW_TIMEOUT_SECONDS'] = budget
        return subprocess.run([VM, *(['--check-shadows'] if shadows else []), *options, path],
                              cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)

    def test_normal_completion_and_nonzero_result(self):
        for value in (0, 7):
            with self.subTest(value=value):
                path = self.module(f'PUSH_I64 {value}\nRET')
                result = self.run_vm(path)
                self.assertEqual(result.returncode == 0, value == 0, result.stderr)
                ordinary = self.run_vm(path, shadows=False)
                self.assertEqual(ordinary.returncode, value)

    def test_assertion_failure(self):
        path = self.module('PUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET')
        result = self.run_vm(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('failed shadow execution', result.stderr)

    def test_deadline_and_invalid_budget(self):
        path = self.module('again:\nJMP again')
        result = self.run_vm(path, budget='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('after 1 seconds', result.stderr)
        result = self.run_vm(path, budget='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('integer from 1 to 300', result.stderr)

    def test_foreign_early_exit_is_not_completion(self):
        source, library = self.directory/'early.c', self.directory/'early.so'
        source.write_text('#include <stdlib.h>\n#include <unistd.h>\n'
                          'void early_exit(void) { exit(0); }\n'
                          'void early_quick_exit(void) { _exit(0); }\n')
        subprocess.run(['cc', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                        '-fPIC', source, '-o', library], check=True, capture_output=True, timeout=30)
        for symbol in ('early_exit', 'early_quick_exit'):
            with self.subTest(symbol=symbol):
                path = self.module('CALL_EXTERN 0\nPUSH_I64 0\nRET',
                                   f'.import "{library}" "{symbol}" void\n.import_kind 0 artifact\n')
                ordinary = self.run_vm(path, shadows=False)
                self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
                result = self.run_vm(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('failed shadow execution', result.stderr)
                self.assertNotIn('could not execute', result.stderr)

    def test_invalid_module_and_incompatible_modes(self):
        invalid = self.directory/'invalid.nvm'
        invalid.write_bytes(b'not a module')
        result = self.run_vm(invalid)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Failed to load', result.stderr)
        path = self.module('PUSH_I64 0\nRET')
        profile = self.directory/'profile.json'
        for options in (('--verify-only',), ('--daemon',), ('--repeat', '2'),
                        ('--profile-isa', str(profile))):
            with self.subTest(options=options):
                result = self.run_vm(path, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('I run shadows once', result.stderr)
        self.assertFalse(profile.exists())
        result = subprocess.run([VM, '--check-shadows', path, '--', 'guest'],
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('I run shadows once', result.stderr)

if __name__ == '__main__': unittest.main()
