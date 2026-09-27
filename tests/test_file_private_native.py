"""I compile actual native functions from the exact private VM fixture corpus."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
import unittest

from tests.test_file_private_vm import PROVIDERS as VM_PROVIDERS

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*VM_PROVIDERS, 'src/nanoisa/nvm2c_file_private.c']


class FilePrivateNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-private-native-'))
        print(f'I retain native artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', ''))
        cls.flags += ['-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_VM_PRIVATE',
                      '-DNVM_FILE_NATIVE_PRIVATE', '-g', '-Wall', '-Wextra', '-Werror',
                      '-I.', '-Isrc', '-Isrc/nanoisa']
        if os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        stems = {Path(source).stem for source in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto -lffi'))
        cls.environment = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS=asan_options("halt_on_error=1"),
                               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        (cls.artifacts / 'environment.json').write_text(json.dumps({k: cls.environment.get(k) for k in
            ('CC', 'NANO_FILE_RUNTIME_CC', 'NANO_FILE_RUNTIME_CFLAGS', 'SDKROOT', 'LSAN_OPTIONS',
             'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'FILE_RUNTIME_OBJECTS', 'FILE_RUNTIME_LDFLAGS')}, indent=2))

    def command(self, name, args, marker=None):
        (self.artifacts / (name + '-command.json')).write_text(json.dumps(args, indent=2))
        stdout_path = self.artifacts / (name + '-stdout.bin')
        stderr_path = self.artifacts / (name + '-stderr.bin')
        started = time.monotonic()
        timed_out = False
        cleanup = []
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            process = subprocess.Popen(args, cwd=ROOT, env=self.environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Regular output files retain all bytes without waiting on pipes
                # held by descendants. Only this fresh process group is signaled.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                        cleanup.append({'signal': sig.name, 'sent': True})
                    except ProcessLookupError:
                        cleanup.append({'signal': sig.name, 'group_absent': True})
                    except OSError as error:
                        cleanup.append({'signal': sig.name, 'error': str(error)})
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        cleanup.append({'signal': sig.name, 'leader_wait_expired': True})
                # KILL is attempted even when TERM reaped the group leader, so
                # a surviving child cannot inherit an indefinite grace period.
        stdout_text = stdout_path.read_bytes().decode('utf-8', errors='replace')
        stderr_text = stderr_path.read_bytes().decode('utf-8', errors='replace')
        status = 124 if timed_out else process.returncode
        terminal = {'timed_out': timed_out, 'timeout_seconds': 240,
                    'elapsed_seconds': round(time.monotonic() - started, 6),
                    'returncode': process.poll(), 'status': status,
                    'cleanup': cleanup, 'leader_reaped': process.returncode is not None}
        (self.artifacts / (name + '.log')).write_text(stdout_text + stderr_text)
        (self.artifacts / (name + '-status.txt')).write_text(str(status) + '\n')
        (self.artifacts / (name + '-terminal.json')).write_text(json.dumps(terminal, indent=2))
        self.assertFalse(timed_out, (args, terminal, stdout_text, stderr_text))
        self.assertEqual(process.returncode, 0, (args, stdout_text, stderr_text))
        if marker:
            self.assertIn(marker, stdout_text)
            print(stdout_text.strip(), flush=True)
        return stdout_text

    def providers(self, name, instrument):
        objects = []
        alloc = ['-include', 'tests/nanoisa/file_runtime_hooks.h', '-Dmalloc=file_test_malloc',
                 '-Dcalloc=file_test_calloc', '-Drealloc=file_test_realloc', '-Dfree=file_test_free']
        for source in PROVIDERS:
            stem = Path(source).stem
            if instrument and stem in ('file_runtime', 'file_vm_private'):
                continue
            hooks = alloc if instrument and stem != 'vm_ffi' else []
            if stem == 'nsi_file' and instrument:
                hooks = [*hooks, '-DFILE_RUNTIME_HOST_HOOKS']
            if stem == 'vm_ffi':
                hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                         '-Dffi_loader_open=file_runtime_loader_open', '-Dfork=file_runtime_fork']
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler, *self.flags, '-O1', *hooks, '-c',
                ('tests/nanoisa/file_runtime_frame_values.c' if instrument and stem == 'nsi_file_values' else source),
                '-o', str(obj)])
            objects.append(str(obj))
        return objects

    def registry(self, name, rows, directory):
        source = ['#include "src/nanoisa/nvm2c_file_private.h"', '#include <stdlib.h>', '#include <string.h>',
                  '#include <stdio.h>']
        for index, status, size in rows:
            if status == 0:
                source.append(f'extern NvmFileRuntimeReport nf_case_{index}(NvmFileRuntimeView *);')
            data = (directory / f'case-{index:03d}.nvm').read_bytes()
            self.assertEqual(len(data), size)
            source.append(f'static const uint8_t wire_{index}[]={{' + ','.join(map(str, data)) + '};')
        source += ['NvmFileRuntimeReport file_native_registered(const uint8_t *bytes,size_t size,NvmFileRuntimeView *out){',
                   'NvmFileRuntimeReport r={0};r.function=r.instruction=UINT32_MAX;',
                   'if(!out){r.status=NVM_FILE_RUNTIME_INVALID;return r;}']
        for index, status, size in rows:
            source.append(f'if(bytes && size=={size} && !memcmp(bytes,wire_{index},size)){{')
            if status == 0:
                source.append(f'return nf_case_{index}(out);}}')
            else:
                # There is deliberately no executable artifact for a refused module.
                source += ['char *text=(char *)(uintptr_t)1;char error[256];',
                           'r.status=nvm2c_file_private_emit(bytes,size,&text,error,sizeof error);',
                           f'if(r.status!={status} || text!=(char *)(uintptr_t)1)abort();', 'return r;}']
        source += ['if(!bytes){r.status=NVM_FILE_RUNTIME_INVALID;return r;}',
                   'fprintf(stderr,"I lack an exact captured native case (%zu bytes)\\n",size);abort();}', '']
        path = self.artifacts / (name + '-registry.c')
        path.write_text('\n'.join(source))
        return path

    def qualify(self, name, instrument):
        objects = self.providers(name, instrument)
        fixture = 'tests/nanoisa/test_file_private_native.c'
        mode = ['-DHOSTED_INSTRUMENT'] if instrument else []
        directory = self.artifacts / (name + '-cases')
        directory.mkdir()
        capture = self.artifacts / (name + '-capture')
        self.command(name + '-capture-build', [*self.compiler, *self.flags, '-O1', *mode,
                     '-DFILE_NATIVE_CAPTURE', fixture, 'tests/nanoisa/test_file_native_buffer.c', *objects, *self.objects, *self.ldflags, '-o', str(capture)])
        self.command(name + '-capture-run', [str(capture), str(directory)], 'PASS native capture:')
        rows = [tuple(map(int, line.split('\t'))) for line in (directory / 'cases.tsv').read_text().splitlines()]
        self.assertGreater(len(rows), 30)
        registry = self.registry(name, rows, directory)
        generated = [(i, directory / f'case-{i:03d}.c') for i, status, _ in rows if status == 0]
        for index, source in generated:
            text = source.read_text()
            self.assertIn('nf_function_0', text)
            self.assertIn('nf_label_0:', text)
            self.assertNotIn('nvm_file_vm_execute', text)
            self.assertNotIn('switch(', text)
            self.assertNotIn('fvm_step', text)
        (self.artifacts / (name + '-generated-sha256.json')).write_text(json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for _, p in generated}, indent=2))
        for opt in ('-O0', '-O2'):
            native_objects = []
            for index, source in generated:
                obj = self.artifacts / f'{name}-{opt[1:]}-{index}.o'
                hooks = ['-include', 'tests/nanoisa/file_native_hooks.h',
                         '-Dnvm_file_runtime_begin=file_native_begin',
                         '-Dnvm_file_runtime_frame_call=file_native_call',
                         '-Dnvm_file_runtime_frame_return=file_native_return',
                         '-Dnvm_file_runtime_service=file_native_service']
                self.command(f'{name}-{opt[1:]}-{index}-build', [*self.compiler, *self.flags, opt,
                    *hooks, f'-Dnvm_file_native_execute=nf_case_{index}', '-c', str(source), '-o', str(obj)])
                native_objects.append(str(obj))
            binary = self.artifacts / f'{name}-{opt[1:]}-replay'
            self.command(f'{name}-{opt[1:]}-link', [*self.compiler, *self.flags, opt, *mode, fixture,
                str(registry), *native_objects, *objects, *self.objects, *self.ldflags, '-o', str(binary)])
            self.command(f'{name}-{opt[1:]}-run', [str(binary)], 'PASS private generated-native replay:')
        if not instrument:
            self.minimal_generated(name, generated[0][1], objects)

    def minimal_generated(self, name, source, objects):
        # This executable has no fixture VM implementation or private VM object.
        kept = [p for p in objects if not p.endswith(('-file_vm_private.o', '-nvm2c_file_private.o'))]
        base = '''#include "src/nanoisa/nvm2c_file_private.h"
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
int g_argc;char **g_argv;
bool file_runtime_loader_init(bool x){(void)x;return false;}
bool file_runtime_loader_open(const char *a,const char *b){(void)a;(void)b;return false;}
pid_t file_runtime_fork(void){return -1;}
NvmFileRuntimeReport nvm_file_native_execute(NvmFileRuntimeView *);
'''
        for kind in ('plain', 'abi', 'facts'):
            unit = self.artifacts / f'{name}-{kind}-isolated.c'
            code = source.read_text()
            if kind == 'abi':
                code = code.replace('nvm_file_runtime_native_abi(1u,', 'nvm_file_runtime_native_abi(2u,')
            if kind == 'facts':
                code, count = re.subn(r'(s\.functions\)!=UINT64_C\()(\d+)',
                                     lambda m: m[1] + str(int(m[2]) + 1), code, count=1)
                self.assertEqual(count, 1)
            unit.write_text(code)
            driver = self.artifacts / f'{name}-{kind}-driver.c'
            if kind == 'plain':
                assertion = 'r.status==NVM_FILE_RUNTIME_OK && r.acquired && out.type.tag==TAG_INT && out.values[0]==INT64_MIN'
            else:
                assertion = 'r.status==NVM_FILE_RUNTIME_UNRESOLVED && !r.acquired && !memcmp(&out,&old,sizeof out)'
            driver.write_text(base + 'int main(void){NvmFileRuntimeView out,old;memset(&out,0xa5,sizeof out);old=out;'
                              '(void)old;NvmFileRuntimeReport r=nvm_file_native_execute(&out);return !(' + assertion + ');}\n')
            for opt in ('-O0', '-O2'):
                exe = self.artifacts / f'{name}-{kind}-{opt[1:]}-isolated'
                self.command(f'{name}-{kind}-{opt[1:]}-isolated-build', [*self.compiler, *self.flags, opt,
                    str(unit), str(driver), *kept, *self.objects, *self.ldflags, '-o', str(exe)])
                self.command(f'{name}-{kind}-{opt[1:]}-isolated-run', [str(exe)])
                symbols = self.command(f'{name}-{kind}-{opt[1:]}-symbols', ['nm', str(exe)])
                self.assertNotIn('nvm_file_vm_execute', symbols)

    def test_instrumented_native_corpus(self):
        self.qualify('instrumented', True)

    def test_linked_native_corpus_and_isolated_refusals(self):
        self.qualify('linked', False)


if __name__ == '__main__':
    unittest.main()
