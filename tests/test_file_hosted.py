"""I qualify nonexecuting serialized File plans and their allocation chain."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.file_runtime_compile_profile import (HOSTED_PROVIDERS as PROVIDERS,
                                               compile_flags, provider_profiles, fixture_profile)

ROOT = Path(__file__).resolve().parents[1]

class FileHosted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-hosted-',
                                            dir=os.environ.get('NANO_FILE_RUNTIME_REPORT_DIR')))
        print(f'I retain hosted-plan artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_HOSTED_CC', 'cc'))
        cls.flags = compile_flags('hosted', shlex.split(os.environ.get('NANO_FILE_HOSTED_CFLAGS', '')),
                                  os.environ.get('NANO_FILE_HOSTED_SANITIZERS', '1') != '0')
        cls.objects = [p for p in shlex.split(os.environ['FILE_HOSTED_OBJECTS'])
                       if Path(p).stem not in PROVIDERS]
        cls.ldflags = shlex.split(os.environ.get('FILE_HOSTED_LDFLAGS', '-lm -lcrypto'))

    def command(self, name, args, run=False):
        (self.artifacts / f'{name}-command.txt').write_text(shlex.join(args) + '\n')
        env = dict(os.environ, ASAN_OPTIONS='halt_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=180)
        (self.artifacts / f'{name}.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, (args, result.stdout, result.stderr))
        if run:
            self.assertIn('PASS', result.stdout)
            print(result.stdout.strip(), flush=True)

    def qualify(self, name, instrument):
        objects = []
        for original, source, hooks in provider_profiles('hosted', instrument):
            provider = Path(original).stem
            obj = self.artifacts / f'{name}-{provider}.o'
            self.command(f'{name}-{provider}-build', [*self.compiler, *self.flags, *hooks,
                         '-c', source, '-o', str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        fixture, options = fixture_profile('hosted', instrument)
        self.command(f'{name}-link', [*self.compiler, *self.flags,
                     *options, fixture, *objects, *self.objects,
                     *self.ldflags, '-o', str(exe)])
        self.command(f'{name}-run', [str(exe)], run=True)

    def test_full_chain_allocation_prefix_and_transient_recovery(self):
        self.qualify('instrumented', True)

    def test_linked_serialized_startup_bounds_and_refusal(self):
        self.qualify('linked', False)

if __name__ == '__main__':
    unittest.main()
