"""I qualify actual macro-gated private File VM dispatch; public routes refuse."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from tests.test_file_hosted import PROVIDERS as HOSTED_PROVIDERS

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*(f'src/nanoisa/{name}.c' for name in HOSTED_PROVIDERS),
             'src/nsi_cap.c', 'src/nsi_file.c', 'src/nsi_file_values.c',
             'src/nanoisa/file_runtime.c', 'src/nanovm/file_vm_private.c', 'src/nanovm/vm_ffi.c']


class FilePrivateVm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-private-vm-'))
        print(f'I retain carrier artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', ''))
        cls.flags += ['-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_VM_PRIVATE', '-g', '-O1', '-Wall',
                      '-Wextra', '-Werror', '-Isrc', '-Isrc/nanoisa']
        if os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        stems = {Path(source).stem for source in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto'))

    def command(self, name, args, run=False):
        (self.artifacts / f'{name}-command.txt').write_text(shlex.join(args) + '\n')
        env = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
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
        alloc = ['-include', 'tests/nanoisa/file_runtime_hooks.h',
                 '-Dmalloc=file_test_malloc', '-Dcalloc=file_test_calloc',
                 '-Drealloc=file_test_realloc', '-Dfree=file_test_free']
        for source in PROVIDERS:
            stem = Path(source).stem
            if instrument and stem in ('file_runtime', 'file_vm_private'):
                # The fixture includes the exact source to inspect sizeof and
                # private accounting without exporting mutable carrier handles.
                continue
            hooks = alloc if instrument and stem != 'vm_ffi' else []
            if stem == 'nsi_file' and instrument:
                hooks = [*hooks, '-DFILE_RUNTIME_HOST_HOOKS']
            if stem == 'vm_ffi':
                # Both modes count attempted public loader/fork calls; these
                # sentinel hooks fail without loading a library or forking.
                hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                         '-Dffi_loader_open=file_runtime_loader_open',
                         '-Dfork=file_runtime_fork']
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler, *self.flags,
                         *hooks, '-c', ('tests/nanoisa/file_runtime_frame_values.c' if instrument and stem == 'nsi_file_values' else source), '-o', str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        self.command(f'{name}-link', [*self.compiler, *self.flags,
                     *(['-DHOSTED_INSTRUMENT'] if instrument else []),
                     'tests/nanoisa/test_file_private_vm.c', *objects, *self.objects,
                     *self.ldflags, '-o', str(exe)])
        self.command(f'{name}-run', [str(exe)], run=True)

    def test_instrumented_private_vm_and_preserved_frames(self):
        self.qualify('instrumented', True)

    def test_linked_private_vm_and_preserved_frames(self):
        self.qualify('linked', False)


if __name__ == '__main__':
    unittest.main()
