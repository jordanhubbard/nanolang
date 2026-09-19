"""I qualify nonexecuting serialized File plans and their allocation chain."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# I rebuild every allocating provider reached by this bounded reader/bridge/plan.
PROVIDERS = [
    'file_flow', 'service_file_nominal_plan', 'service_bindings_module',
    'nvm_format', 'nvm_v2_constants', 'nvm_v2_signatures', 'nvm_v2_layouts',
    'nvm_v2_functions', 'nvm_v2_imports', 'nvm_v2_module',
    'nvm_v2_convert', 'retained_layouts', 'ownership_contracts',
]

class FileHosted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-hosted-'))
        print(f'I retain hosted-plan artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_HOSTED_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_FILE_HOSTED_CFLAGS', ''))
        cls.flags += ['-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra', '-Werror']
        if os.environ.get('NANO_FILE_HOSTED_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        cls.objects = [p for p in shlex.split(os.environ['FILE_HOSTED_OBJECTS'])
                       if Path(p).stem not in PROVIDERS]
        cls.ldflags = shlex.split(os.environ.get('FILE_HOSTED_LDFLAGS', '-lm -lcrypto'))

    def command(self, name, args, run=False):
        (self.artifacts / f'{name}-command.txt').write_text(shlex.join(args) + '\n')
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
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
        hooks = (['-include', 'tests/nanoisa/file_hosted_alloc.h',
                  '-Dmalloc=file_test_malloc', '-Dcalloc=file_test_calloc',
                  '-Drealloc=file_test_realloc', '-Dfree=file_test_free'] if instrument else [])
        for provider in PROVIDERS:
            obj = self.artifacts / f'{name}-{provider}.o'
            self.command(f'{name}-{provider}-build', [*self.compiler, *self.flags, *hooks,
                         '-c', f'src/nanoisa/{provider}.c', '-o', str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        self.command(f'{name}-link', [*self.compiler, *self.flags,
                     *(['-DHOSTED_INSTRUMENT'] if instrument else []),
                     'tests/nanoisa/test_file_hosted.c', *objects, *self.objects,
                     *self.ldflags, '-o', str(exe)])
        self.command(f'{name}-run', [str(exe)], run=True)

    def test_full_chain_allocation_prefix_and_transient_recovery(self):
        self.qualify('instrumented', True)

    def test_linked_serialized_startup_bounds_and_refusal(self):
        self.qualify('linked', False)

if __name__ == '__main__':
    unittest.main()
