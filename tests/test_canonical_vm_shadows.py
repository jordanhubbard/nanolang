"""I check canonical bytecode shadows without a native shadow compiler."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT/'bin/nanoc_stage2')).resolve()

class CanonicalVMShadows(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='canonical-vm-shadows-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source, self.output = self.directory/'input.nano', self.directory/'output.nvm'
        self.output.write_bytes(b'prior artifact')
        self.environment = dict(os.environ)
        self.environment.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)
        self.environment['NANO_VM'] = str(ROOT/'bin/nano_vm')
        self.marker = self.directory/'native-compiler-called'
        cc = self.directory/'reject-cc'
        cc.write_text('#!/bin/sh\n: > ' + shlex.quote(str(self.marker)) + '\nexit 91\n')
        cc.chmod(0o755)
        self.environment['NANO_CC'] = str(cc)
        self.environment['CC'] = str(cc)

    def compile(self, source, *options, success=True):
        self.source.write_text(source)
        result = subprocess.run([COMPILER, self.source, '--emit-nvm', '-o', self.output, *options],
                                cwd=ROOT, env=self.environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertFalse(self.marker.exists(), 'I invoked the native compiler')
        if not success: self.assertEqual(self.output.read_bytes(), b'prior artifact')
        return result

    def test_shared_state_order_and_ordinary_main(self):
        self.compile('''let mut state: int = 0
fn step() -> int { set state (+ state 1) return state }
shadow step { assert (== (step) 1) }
shadow step { assert (== (step) 2) }
fn main() -> int { return 0 }
shadow main { assert (== state 2) assert (== (main) 0) }
''')
        result = subprocess.run([ROOT/'bin/nano_vm', self.output], capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_selection(self):
        dependency = self.directory/'dependency.nano'
        dependency.write_text('module Dep\npub fn value() -> int { return 7 }\nshadow value { assert false }\n')
        source = f'''module "{dependency}" as dep
fn main() -> int {{ return (- (dep.value) 7) }}
shadow main {{ assert (== (main) 0) }}
'''
        failed = self.compile(source, success=False)
        self.assertIn('after failed shadows', failed.stdout + failed.stderr)
        self.compile(source, '--root-shadows-only')

    def test_failure_and_deadline_preserve_previous_output(self):
        for body in ('assert false', 'while true {}'):
            with self.subTest(body=body):
                self.environment['NANO_SHADOW_TIMEOUT_SECONDS'] = '1'
                result = self.compile('fn main() -> int { return 0 }\nshadow main { ' + body + ' }\n', success=False)
                self.assertIn('after failed shadows', result.stdout + result.stderr)
                if body.startswith('while'): self.assertIn('after 1 seconds', result.stderr)

    def test_missing_vm_preserves_output_and_no_shadow_needs_no_runner(self):
        self.environment['NANO_VM'] = str(self.directory/'absent-vm')
        result = self.compile('fn main() -> int { return 0 }\nshadow main { assert true }\n', success=False)
        self.assertIn('after failed shadows', result.stdout + result.stderr)
        self.compile('fn main() -> int { return 0 }\n')

    def test_runner_path_with_spaces(self):
        wrapper = self.directory/'vm runner'
        marker = self.directory/'vm-called'
        wrapper.write_text('#!/bin/sh\n: > ' + shlex.quote(str(marker)) + '\nexec ' +
                           shlex.quote(str(ROOT/'bin/nano_vm')) + ' "$@"\n')
        wrapper.chmod(0o755)
        self.environment['NANO_VM'] = str(wrapper)
        self.compile('fn main() -> int { return 0 }\nshadow main { assert true }\n')
        self.assertTrue(marker.exists())

if __name__ == '__main__': unittest.main()
