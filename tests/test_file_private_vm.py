"""I qualify actual macro-gated private File VM dispatch; public routes refuse."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from tests.file_runtime_compile_profile import (HOSTED_PROVIDERS, VM_PROVIDERS as PROVIDERS,
                                               compile_flags, provider_profiles, fixture_profile)

ROOT = Path(__file__).resolve().parents[1]


class FilePrivateVm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-private-vm-',
                                            dir=os.environ.get('NANO_FILE_RUNTIME_REPORT_DIR')))
        print(f'I retain carrier artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = compile_flags('vm', shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', '')),
                                  os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0')
        stems = {Path(source).stem for source in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto'))

    def command(self, name, args, run=False):
        (self.artifacts / f'{name}-command.txt').write_text(shlex.join(args) + '\n')
        env = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS='halt_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=180)
        (self.artifacts / f'{name}.log').write_text(result.stdout + result.stderr)
        (self.artifacts / f'{name}-status.txt').write_text(str(result.returncode) + '\n')
        self.assertEqual(result.returncode, 0, (args, result.stdout, result.stderr))
        if run:
            self.assertIn('manual private File carrier checks', result.stdout)
            self.assertIn('no File CODE/frame dispatch', result.stdout)
            self.assertIn('manual private File frame transfer checks; no opcode dispatcher', result.stdout)
            self.assertIn('actual private File VM dispatch checks', result.stdout)
            print(result.stdout.strip(), flush=True)

    def qualify(self, name, instrument):
        objects = []
        for original, source, hooks in provider_profiles('vm', instrument):
            stem = Path(original).stem
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler, *self.flags,
                         *hooks, '-c', source, '-o', str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        fixture, options = fixture_profile('vm', instrument)
        self.command(f'{name}-link', [*self.compiler, *self.flags,
                     *options, fixture, *objects, *self.objects,
                     *self.ldflags, '-o', str(exe)])
        self.command(f'{name}-run', [str(exe)], run=True)

    def test_instrumented_private_vm_and_preserved_frames(self):
        self.qualify('instrumented', True)

    def test_linked_private_vm_and_preserved_frames(self):
        self.qualify('linked', False)


if __name__ == '__main__':
    unittest.main()
