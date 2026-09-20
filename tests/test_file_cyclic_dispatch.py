"""I compare private cyclic VM execution with real generated native functions."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
import unittest

from tests.test_file_cyclic_runtime import PROVIDERS as CARRIER_PROVIDERS
from tests import test_file_cyclic as cyclic_runner

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*CARRIER_PROVIDERS, 'src/nanovm/file_vm_cyclic_private.c',
             'src/nanoisa/nvm2c_file_cyclic_private.c']


class FileCyclicDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-cyclic-dispatch-'))
        print(f'I retain cyclic dispatch artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', '')),
                     '-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_CYCLIC_VM_PRIVATE',
                     '-DNVM_FILE_CYCLIC_NATIVE_PRIVATE', '-g', '-Wall', '-Wextra',
                     '-Werror', '-I.', '-Isrc', '-Isrc/nanoisa']
        if os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        stems = {Path(p).stem for p in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.native_base = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_CYCLIC_NATIVE_LINK_OBJECTS'])
                                            if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto -lffi'))
        cls.hooks = ['-include', 'tests/nanoisa/file_cyclic_dispatch_hooks.h',
                     '-Dnvm_file_runtime_begin=dispatch_begin',
                     '-Dnvm_file_runtime_frame_call=dispatch_call',
                     '-Dnvm_file_runtime_frame_return=dispatch_return',
                     '-Dnvm_file_runtime_service=dispatch_service',
                     '-Dnvm_file_runtime_cyclic_destroy=dispatch_destroy']
        (cls.artifacts / 'inputs.json').write_text(json.dumps({
            'compiler': cls.compiler, 'flags': cls.flags, 'providers': PROVIDERS,
            'ordinary_objects': cls.objects, 'native_only_objects': cls.native_base,
            'link_flags': cls.ldflags, 'SDKROOT': os.environ.get('SDKROOT'),
            'LSAN_OPTIONS': '', 'instrumentation': 'rebuilt listed providers and generated C; common objects retain their recorded build'}, indent=2))

    # The retained query runner writes files on launch/wait failure, bounds
    # TERM/KILL after normal and timed-out completion, and checks group exit.
    def command(self, name, args):
        return cyclic_runner.FileCyclic.command(self, name, args)

    def providers(self, name, instrument):
        objects = []
        alloc = ['-include', 'tests/nanoisa/file_runtime_hooks.h',
                 '-Dmalloc=file_test_malloc', '-Dcalloc=file_test_calloc',
                 '-Drealloc=file_test_realloc', '-Dfree=file_test_free']
        for source in PROVIDERS:
            stem = Path(source).stem
            if instrument and stem in ('file_runtime', 'file_vm_cyclic_private'):
                continue
            hooks = alloc if instrument and stem != 'vm_ffi' else []
            if stem == 'nsi_file':
                if not instrument:
                    hooks = ['-include', 'tests/nanoisa/file_runtime_hooks.h']
                hooks += ['-DFILE_RUNTIME_HOST_HOOKS']
            if stem == 'vm_ffi':
                hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                         '-Dffi_loader_open=file_runtime_loader_open', '-Dfork=file_runtime_fork']
            if stem == 'file_vm_cyclic_private':
                hooks += self.hooks
            actual = ('tests/nanoisa/file_cyclic_dispatch_values.c'
                      if instrument and stem == 'nsi_file_values' else source)
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler, *self.flags, '-O1', *hooks,
                                                '-c', actual, '-o', str(obj)])
            objects.append(str(obj))
        return objects

    def registry(self, name, rows, directory):
        lines = ['#include "src/nanoisa/nvm2c_file_cyclic_private.h"', '#include <stdlib.h>', '#include <string.h>']
        for index, status, size in rows:
            if status == 0:
                lines.append(f'extern NvmFileCyclicExecutionReport nf_case_{index}(const NvmFileCyclicOptions *,NvmFileRuntimeView *);')
            data = (directory / f'case-{index:03d}.nvm').read_bytes()
            self.assertEqual(len(data), size)
            lines.append(f'static const uint8_t wire_{index}[]={{' + ','.join(map(str, data)) + '};')
        lines += ['NvmFileCyclicExecutionReport file_cyclic_registered(const uint8_t *bytes,size_t size,const NvmFileCyclicOptions *options,NvmFileRuntimeView *out){',
                  'NvmFileCyclicExecutionReport r={0};r.revision=1;r.instruction_limit=options?options->instruction_limit:0;r.runtime.function=r.runtime.instruction=UINT32_MAX;']
        for index, status, size in rows:
            lines.append(f'if(bytes && size=={size} && !memcmp(bytes,wire_{index},size)){{')
            if status == 0:
                lines.append(f'return nf_case_{index}(options,out);}}')
            else:
                lines += ['char *text=(char *)(uintptr_t)1;char error[256];',
                          'r.runtime.status=nvm2c_file_cyclic_private_emit(bytes,size,&text,error,sizeof error);',
                          f'if(r.runtime.status!={status} || text!=(char *)(uintptr_t)1)abort();\nreturn r;}}']
        lines += ['abort();}', '']
        result = self.artifacts / f'{name}-registry.c'
        result.write_text('\n'.join(lines))
        return result

    def qualify(self, name, instrument):
        objects = self.providers(name, instrument)
        fixture = 'tests/nanoisa/test_file_cyclic_dispatch.c'
        mode = ['-DHOSTED_INSTRUMENT'] if instrument else []
        directory = self.artifacts / f'{name}-cases'
        directory.mkdir()
        capture = self.artifacts / f'{name}-capture'
        self.command(f'{name}-capture-build', [*self.compiler, *self.flags, '-O1', *mode,
            '-DFILE_CYCLIC_CAPTURE', fixture, 'tests/nanoisa/test_file_cyclic_native_buffer.c',
            *objects, *self.objects, *self.ldflags, '-o', str(capture)])
        output = self.command(f'{name}-capture-run', [str(capture), str(directory)])
        self.assertIn(b'PASS cyclic VM capture:', output)
        reference = [line for line in output.splitlines() if line.startswith(b'TRACE ')]
        self.assertGreaterEqual(len(reference), 28)
        rows = [tuple(map(int, row.split('\t'))) for row in (directory / 'cases.tsv').read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 15)
        registry = self.registry(name, rows, directory)
        generated = [(i, directory / f'case-{i:03d}.c') for i, status, _ in rows if status == 0]
        for _, source in generated:
            text = source.read_text()
            self.assertIn('nf_function_', text)
            self.assertIn('nf_label_0:', text)
            self.assertIn('nvm_file_runtime_cyclic_enter(c)', text)
            self.assertNotIn('nvm_file_vm_cyclic_execute', text)
            self.assertNotIn('fvm_step', text)
            # switch(n) compares reference slots only in nf_agrees, not opcodes.
            self.assertNotIn('switch(', text[text.index('static NvmFileRuntimeStatus nf_function_'):])
        (self.artifacts / f'{name}-generated-sha256.json').write_text(json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for _, p in generated}, indent=2))
        for opt in ('-O0', '-O2'):
            compiled = []
            for index, source in generated:
                obj = self.artifacts / f'{name}-{opt[1:]}-{index}.o'
                self.command(f'{name}-{opt[1:]}-{index}-build', [*self.compiler, *self.flags, opt,
                    *self.hooks, f'-Dnvm_file_native_cyclic_execute=nf_case_{index}',
                    '-c', str(source), '-o', str(obj)])
                compiled.append(str(obj))
            binary = self.artifacts / f'{name}-{opt[1:]}-replay'
            self.command(f'{name}-{opt[1:]}-link', [*self.compiler, *self.flags, opt, *mode,
                fixture, str(registry), *compiled, *objects, *self.objects, *self.ldflags, '-o', str(binary)])
            replay = self.command(f'{name}-{opt[1:]}-run', [str(binary)])
            self.assertIn(b'PASS cyclic native replay:', replay)
            self.assertEqual(reference, [line for line in replay.splitlines() if line.startswith(b'TRACE ')])
        if not instrument:
            self.isolated(name, generated, objects)

    def isolated(self, name, generated, objects):
        # The exact query/core closure excludes VM, FFI, private dispatcher and emitter objects.
        kept = [p for p in objects if Path(p).name not in {
            f'{name}-vm_ffi.o', f'{name}-file_vm_cyclic_private.o', f'{name}-nvm2c_file_cyclic_private.o'}]
        paths = [*kept, *self.native_base]
        self.assertFalse(any(Path(p).name in ('vm.o', 'file_vm_cyclic_private.o', 'file_public_vm.o') for p in paths))
        (self.artifacts / f'{name}-isolated-objects.json').write_text(json.dumps(paths, indent=2))
        text = generated[0][1].read_text()
        alternatives = next(p.read_text() for _, p in generated if re.search(r'hosted_variant\(p,\d+,\d+,1,&variant\)', p.read_text()))
        references = next(p.read_text() for _, p in generated if 'reference.owner)!=UINT64_C(' in p.read_text())
        controls = [('plain', text, 0), ('abi_revision', text.replace('cyclic_native_abi(1u,', 'cyclic_native_abi(2u,'), 4),
                    ('abi_size', text.replace('sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport)',
                                             'sizeof(NvmFileCyclicOptions)+1,sizeof(NvmFileCyclicExecutionReport)'), 4)]
        def alter_once(code, pattern):
            changed, count = re.subn(pattern, lambda m: m[1] + str(int(m[2]) + 1), code, count=1)
            self.assertEqual(count, 1)
            return changed
        split = re.search(r'hosted_variant\(p,\d+,\d+,1,&variant\)', alternatives).end()
        controls.append(('later_variant', alternatives[:split] + alter_once(alternatives[split:], r'(variant\.input\.owners\)!=UINT64_C\()(\d+)'), 4))
        controls.append(('reference', alter_once(references, r'(reference\.owner\)!=UINT64_C\()(\d+)'), 4))
        controls.append(('edge', alter_once(alternatives, r'(variant\.edge_variants\[0\]\)!=UINT64_C\()(\d+)'), 4))
        self.assertIn('nf_label_2:\nreturn nf_bad(c);', text)
        controls.append(('dead_label', text.replace('\ngoto nf_label_0;\n', '\ngoto nf_label_2;\n', 1), 3))
        base = '''#include "src/nanoisa/nvm2c_file_cyclic_private.h"
#include <stdio.h>
#include <string.h>
int g_argc;char **g_argv;
static unsigned opens,closes;
FILE *file_runtime_tmpfile(void){opens++;return tmpfile();}
int file_runtime_fclose(FILE *f){closes++;return fclose(f);}
size_t file_runtime_fread(void *p,size_t n,size_t c,FILE *f){return fread(p,n,c,f);}
size_t file_runtime_fwrite(const void *p,size_t n,size_t c,FILE *f){return fwrite(p,n,c,f);}
int file_runtime_fseek(FILE *f,long o,int w){return fseek(f,o,w);}
int file_runtime_ferror(FILE *f){return ferror(f);}
'''
        for kind, code, expectation in controls:
            source = self.artifacts / f'{name}-{kind}-isolated.c'
            source.write_text(code)
            driver = self.artifacts / f'{name}-{kind}-driver.c'
            if kind == 'plain':
                condition = 'r.runtime.status==NVM_FILE_RUNTIME_OK && out.type.tag==TAG_INT && out.values[0]==17 && r.instructions_started==2'
            elif kind == 'dead_label':
                condition = 'r.runtime.status==NVM_FILE_RUNTIME_STATE && r.runtime.acquired && !r.instructions_started && !memcmp(&out,&old,sizeof out)'
            else:
                condition = 'r.runtime.status==NVM_FILE_RUNTIME_UNRESOLVED && !r.runtime.acquired && !r.instructions_started && !memcmp(&out,&old,sizeof out)'
            driver.write_text(base + 'int main(void){NvmFileCyclicOptions options={1,100000};NvmFileRuntimeView out,old;memset(&out,0xa5,sizeof out);old=out;(void)old;NvmFileCyclicExecutionReport r=nvm_file_native_cyclic_execute(&options,&out);return !(' + condition + ' && r.revision==1 && r.instruction_limit==100000 && !r.fuel_exhausted && !opens && !closes);}\n')
            for opt in ('-O0', '-O2'):
                exe = self.artifacts / f'{name}-{kind}-{opt[1:]}'
                self.command(f'{name}-{kind}-{opt[1:]}-build', [*self.compiler, *self.flags, opt,
                    str(source), str(driver), *paths, *self.ldflags, '-o', str(exe)])
                self.command(f'{name}-{kind}-{opt[1:]}-run', [str(exe)])
                symbols = self.command(f'{name}-{kind}-{opt[1:]}-symbols', ['nm', str(exe)])
                self.assertNotIn(b'nvm_file_vm_cyclic_execute', symbols)
                self.assertNotRegex(symbols, rb'\b_?(vm_execute|vm_core_execute|nvm_file_vm_execute)\b')

    def test_instrumented_matched_cyclic_dispatch(self):
        self.qualify('instrumented', True)

    def test_linked_matched_dispatch_and_isolated_refusals(self):
        self.qualify('linked', False)


if __name__ == '__main__':
    unittest.main()
