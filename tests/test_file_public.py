"""I qualify grant-held public VM/native APIs and the installed File package."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import tempfile
import time
import unittest

from tests.test_file_private_vm import PROVIDERS as VM_PROVIDERS

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*VM_PROVIDERS, 'src/nanoisa/nvm2c_file_private.c',
             'src/nanoisa/file_public_native.c', 'src/nanovm/file_public_vm.c',
             'src/nanoisa/file_host_grant.c']


class FilePublic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-public-'))
        print(f'I retain native artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', ''))
        cls.flags += ['-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_VM_PRIVATE',
                      '-DNVM_FILE_NATIVE_PRIVATE', '-DNVM_FILE_PUBLIC_ENGINE', '-g', '-Wall', '-Wextra', '-Werror',
                      '-I.', '-Isrc', '-Isrc/nanoisa', '-pthread']
        if os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        stems = {Path(source).stem for source in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto -lffi'))
        cls.environment = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        (cls.artifacts / 'environment.json').write_text(json.dumps({k: cls.environment.get(k) for k in
            ('CC', 'NANO_FILE_RUNTIME_CC', 'NANO_FILE_RUNTIME_CFLAGS', 'SDKROOT', 'LSAN_OPTIONS',
             'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'FILE_RUNTIME_OBJECTS', 'FILE_RUNTIME_LDFLAGS')}, indent=2))

    @staticmethod
    def stop_group(process):
        outcome = {'pid':process.pid,'errors':[],'reaped':False,'group_gone':False}
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:
                os.killpg(process.pid,sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                outcome['errors'].append(repr(error))
            try:
                process.wait(timeout=5)
                outcome['reaped']=True
            except subprocess.TimeoutExpired:
                outcome[sig.name + '_wait_expired']=True
        until=time.monotonic()+2
        while True:
            try:
                os.killpg(process.pid,0)
            except ProcessLookupError:
                outcome['group_gone']=True
                break
            except OSError as error:
                outcome['errors'].append(repr(error))
                break
            if time.monotonic()>=until:
                break
            time.sleep(0.02)
        outcome['confirmed']=outcome['reaped'] and outcome['group_gone'] and not outcome['errors']
        return outcome

    def command(self, name, args, marker=None, expected=0, cwd=None, timeout=240, stdout_fd=None):
        (self.artifacts / (name + '-command.json')).write_text(json.dumps(args,indent=2))
        stdout_path=self.artifacts / (name + '-stdout.bin')
        stderr_path=self.artifacts / (name + '-stderr.bin')
        started=time.monotonic()
        process=None
        problem=None
        cleanup=None
        try:
            with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                process=subprocess.Popen(args,cwd=cwd or ROOT,env=self.environment,
                    stdin=subprocess.DEVNULL,stdout=stdout if stdout_fd is None else stdout_fd,
                    stderr=stderr,start_new_session=True)
                (self.artifacts / (name + '-process.json')).write_text(json.dumps(
                    {'pid':process.pid,'pgid':process.pid}))
                process.wait(timeout=timeout)
        except BaseException as error:
            problem=error
        finally:
            if process is not None:
                cleanup=self.stop_group(process)
            timed_out=isinstance(problem,(subprocess.TimeoutExpired,TimeoutError))
            status=124 if timed_out else (127 if process is None else process.returncode)
            terminal={'timed_out':timed_out,'timeout_seconds':timeout,
                'elapsed_seconds':round(time.monotonic()-started,6),
                'returncode':None if process is None else process.returncode,'status':status,
                'cleanup':cleanup,'exception':None if problem is None else repr(problem)}
            (self.artifacts / (name + '-status.txt')).write_text(str(status)+'\n')
            (self.artifacts / (name + '-terminal.json')).write_text(json.dumps(terminal,indent=2))
            stdout_text=stdout_path.read_bytes().decode('utf-8',errors='replace') if stdout_path.exists() else ''
            stderr_text=stderr_path.read_bytes().decode('utf-8',errors='replace') if stderr_path.exists() else ''
            (self.artifacts / (name + '.log')).write_text(stdout_text+stderr_text)
        if problem is not None:
            raise problem
        self.assertTrue(cleanup and cleanup['confirmed'],(args,terminal))
        self.assertEqual(process.returncode,expected,(args,stdout_text,stderr_text))
        if marker:
            self.assertIn(marker,stdout_text)
            print(stdout_text.strip(),flush=True)
        return stdout_text

    def providers(self, name, instrument):
        objects = []
        alloc = ['-include', 'tests/nanoisa/file_runtime_hooks.h', '-Dmalloc=file_test_malloc',
                 '-Dcalloc=file_test_calloc', '-Drealloc=file_test_realloc', '-Dfree=file_test_free']
        for source in PROVIDERS:
            stem = Path(source).stem
            if instrument and stem in ('file_runtime', 'file_vm_private'):
                continue
            hooks = alloc if instrument and stem not in ('vm_ffi','file_host_grant') else []
            if stem == 'file_public_vm':
                hooks = [*hooks, '-include', 'tests/nanoisa/file_native_hooks.h',
                         '-Dnvm_file_runtime_begin=file_native_begin',
                         '-Dnvm_file_runtime_frame_call=file_native_call',
                         '-Dnvm_file_runtime_frame_return=file_native_return',
                         '-Dnvm_file_runtime_service=file_native_service']
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
        if not instrument:
            reference = self.artifacts / (name + '-qualified-reference.c')
            old = self.command(name + '-reference-source', ['git','show',
                'f1606e2c84e67491e9652a5bf71944d235216d95:src/nanoisa/nvm2c_file_private.c'])
            reference.write_text(old)
            obj = self.artifacts / (name + '-qualified-reference.o')
            self.command(name + '-reference-build', [*self.compiler,*self.flags,'-O1',
                '-Dnvm2c_file_private_emit=file_reference_emit','-c',str(reference),'-o',str(obj)])
            objects.append(str(obj))
        return objects

    def registry(self, name, rows, directory):
        source = ['#include "src/nanoisa/file_public.h"', '#include <stdlib.h>',
                  '#include <string.h>', '#include <stdio.h>',
                  'NvmFileHostGrant *file_public_test_grant(void);']
        for index, status, size in rows:
            if status == 0:
                source.append(f'extern NvmFileRuntimeReport nf_case_{index}(NvmFileHostGrant *,NvmFileScalar *);')
            data = (directory / f'case-{index:03d}.nvm').read_bytes()
            self.assertEqual(len(data), size)
            source.append(f'static const uint8_t wire_{index}[]={{' + ','.join(map(str,data)) + '};')
        first = next(i for i,status,_ in rows if status == 0)
        source += ['void file_public_native_busy_check(void){NvmFileScalar out,old;',
                   'memset(&out,0xa5,sizeof out);old=out;',
                   f'NvmFileRuntimeReport r=nf_case_{first}(file_public_test_grant(),&out);',
                   'if(r.status!=NVM_FILE_RUNTIME_BUSY || r.acquired || memcmp(&out,&old,sizeof out))abort();}']
        source += ['NvmFileRuntimeReport file_native_registered(const uint8_t *bytes,size_t size,NvmFileRuntimeView *out){',
                   'NvmFileRuntimeReport r={0};r.function=r.instruction=UINT32_MAX;',
                   'if(!out){r.status=NVM_FILE_RUNTIME_INVALID;return r;}']
        for index, status, size in rows:
            source.append(f'if(bytes && size=={size} && !memcmp(bytes,wire_{index},size)){{')
            if status == 0:
                source += ['NvmFileScalar scalar,old;memset(&scalar,0xa5,sizeof scalar);old=scalar;',
                           f'r=nf_case_{index}(file_public_test_grant(),&scalar);',
                           'if(r.status==NVM_FILE_RUNTIME_OK){NvmFileRuntimeView v={0};v.initialized=true;v.fields=1;',
                           'v.type.tag=scalar.tag;v.type.category=NVM_FILE_CATEGORY_UNKNOWN;',
                           'v.type.global_index=v.type.catalog_ordinal=UINT32_MAX;v.values[0]=scalar.value;*out=v;}',
                           'else if(memcmp(&scalar,&old,sizeof scalar)){abort();}', 'return r;}']
            else:
                source += ['char *text=(char *)(uintptr_t)1;char error[256];',
                           'r.status=nvm2c_emit_file_bytes(bytes,size,"refused",&text,error,sizeof error);',
                           f'if(r.status!={status} || text!=(char *)(uintptr_t)1){{abort();}}', 'return r;}']
        source += ['if(!bytes){r.status=NVM_FILE_RUNTIME_INVALID;return r;}',
                   'fprintf(stderr,"I lack an exact public captured case (%zu bytes)\\n",size);abort();}', '']
        path = self.artifacts / (name + '-registry.c')
        path.write_text('\n'.join(source))
        return path

    def qualify(self, name, instrument):
        self.prepare_package(name)
        objects = self.providers(name, instrument)
        fixture = 'tests/nanoisa/test_file_public.c'
        mode = ['-DHOSTED_INSTRUMENT'] if instrument else []
        directory = self.artifacts / (name + '-cases')
        directory.mkdir()
        capture = self.artifacts / (name + '-capture')
        self.command(name + '-capture-build', [*self.compiler, *self.flags, '-O1', *mode,
                     '-DFILE_NATIVE_CAPTURE', fixture, 'tests/nanoisa/test_file_native_buffer.c', *objects, *self.objects, *self.ldflags, '-o', str(capture)])
        self.command(name + '-capture-run', [str(capture), str(directory)], 'PASS public VM corpus, shared grant and scalar boundary')
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
                self.command(f'{name}-{opt[1:]}-{index}-build', [*self.compiler, *self.flags, opt, '-std=c99',
                    '-I' + str(self.installed / 'include'), *hooks, f'-Dnvm_file_program_case=nf_case_{index}', '-c', str(source), '-o', str(obj)])
                native_objects.append(str(obj))
            binary = self.artifacts / f'{name}-{opt[1:]}-replay'
            self.command(f'{name}-{opt[1:]}-link', [*self.compiler, *self.flags, opt, *mode, fixture,
                str(registry), *native_objects, *objects, *self.objects, *self.ldflags, '-o', str(binary)])
            self.command(f'{name}-{opt[1:]}-run', [str(binary)], 'PASS public generated-native corpus and shared gate')
        if not instrument:
            self.private_parity(name, rows, directory, objects)
            self.installed_controls(name, generated[0][1], directory, rows)

    def prepare_package(self, name):
        if hasattr(self, 'installed'):
            return
        type(self).installed = self.artifacts / 'installed'
        # Every compiler/configuration receives a fresh owning C11 object and
        # archive. Consumers use C99; the Make owner recipe appends C11 itself.
        flags = [f for f in self.flags if f not in
                 ('-DNVM_FILE_VM_PRIVATE','-DNVM_FILE_NATIVE_PRIVATE','-DNVM_FILE_PUBLIC_ENGINE')]
        flags += ['-std=c99','-fPIC','-D_GNU_SOURCE']
        install_log = self.command(name + '-fresh-install', ['make','-B','-j2','install',
            'PREFIX=' + str(self.installed), 'CC=' + shlex.join(self.compiler),
            'CFLAGS=' + shlex.join(flags), 'LDFLAGS=' + shlex.join(self.ldflags)], timeout=900)
        grant_lines=[line for line in install_log.splitlines() if ' -c src/nanoisa/file_host_grant.c ' in line]
        self.assertTrue(grant_lines)
        self.assertTrue(all('-std=c11' in line and line.rfind('-std=c11')>line.rfind('-std=c99') for line in grant_lines))
        paths = [p for p in self.installed.rglob('*') if p.is_file()]
        self.assertTrue((self.installed / 'lib/libnano_file_runtime.a').is_file())
        self.assertEqual(len(list((self.installed / 'include').rglob('*.h'))),30)
        (self.artifacts / 'installed-sha256.json').write_text(json.dumps(
            {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},indent=2))
        self.command(name + '-archive-members', ['ar','t',str(self.installed / 'lib/libnano_file_runtime.a')])

    def private_parity(self, name, rows, directory, objects):
        del rows, objects
        current = sorted(directory.glob('parity-*-current.c'))
        self.assertGreater(len(current),30)
        pairs = []
        for path in current:
            reference = path.with_name(path.name.replace('-current.c','-reference.c'))
            self.assertEqual(path.read_bytes(),reference.read_bytes())
            pairs.append({'current':str(path),'reference':str(reference),
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.artifacts / (name + '-private-byte-parity.json')).write_text(json.dumps(pairs,indent=2))

    def installed_controls(self, name, source, directory, rows):
        del source, rows
        outside = self.artifacts / 'outside repository'
        outside.mkdir()
        wire = directory / 'public-lifecycle.nvm'
        vm = self.installed / 'bin/nano_vm'
        emitter = self.installed / 'bin/nvm2c'
        self.command(name + '-installed-vm', [str(vm),'--allow-temporary-files',str(wire)],
                     expected=251,cwd=outside)
        for label,status in (('bool-false',0),('bool-true',1),('negative-int',255)):
            self.command(name + '-installed-' + label,[str(vm),'--allow-temporary-files',
                str(directory / (label + '.nvm'))],expected=status,cwd=outside)
        self.command(name + '-installed-default-refusal', [str(vm),str(wire)],expected=1,cwd=outside)
        for index,option in enumerate(('--verify-only','--daemon','--check-shadows','--debug','--cop')):
            self.command(name + f'-installed-mode-{index}', [str(vm),'--allow-temporary-files',option,str(wire)],
                         expected=1,cwd=outside)
        for index,options in enumerate((['--repeat','1'],['--profile-isa',str(outside / 'profile.json')])):
            self.command(name + f'-installed-argument-mode-{index}',[str(vm),'--allow-temporary-files',*options,str(wire)],expected=1,cwd=outside)
        self.command(name + '-installed-guest-refusal',[str(vm),'--allow-temporary-files',str(wire),'--','guest'],expected=1,cwd=outside)
        # File C output publication must preserve a preexisting destination on
        # admission/identifier/read failures, and leave no staged temp file.
        output = outside / 'generated.c'
        output.write_bytes(b'preserved output\n')
        self.command(name + '-missing-opt-in', [str(emitter),str(wire),'-o',str(output)],expected=1,cwd=outside)
        self.assertEqual(output.read_bytes(),b'preserved output\n')
        self.command(name + '-invalid-name', [str(emitter),'--file-temporary','--entry-name','bad-name',str(wire),'-o',str(output)],expected=1,cwd=outside)
        self.assertEqual(output.read_bytes(),b'preserved output\n')
        self.command(name + '-missing-entry-name',[str(emitter),'--file-temporary',str(wire),'-o',str(output)],expected=2,cwd=outside)
        self.command(name + '-missing-file-option',[str(emitter),'--entry-name','name',str(wire),'-o',str(output)],expected=2,cwd=outside)
        self.assertEqual(output.read_bytes(),b'preserved output\n')
        empty = outside / 'empty.nvm';empty.write_bytes(b'')
        self.command(name + '-empty-input', [str(emitter),'--file-temporary','--entry-name','empty',str(empty),'-o',str(output)],expected=1,cwd=outside)
        self.assertEqual(output.read_bytes(),b'preserved output\n')
        large = outside / 'large.nvm'
        with large.open('wb') as stream:
            stream.truncate(16*1024*1024+1)
        self.command(name + '-large-input', [str(emitter),'--file-temporary','--entry-name','large',str(large),'-o',str(output)],expected=1,cwd=outside)
        self.assertEqual(output.read_bytes(),b'preserved output\n')
        sources = []
        for ident in ('first','second'):
            target = outside / (ident + '.c')
            self.command(name + '-emit-' + ident, [str(emitter),'--file-temporary','--entry-name',ident,str(wire),'-o',str(target)],cwd=outside)
            text = target.read_text()
            self.assertIn('nvm_file_program_' + ident,text)
            for forbidden in ('nvm_file_vm_execute','file_vm_execute_serialized','fvm_step','switch('):
                self.assertNotIn(forbidden,text)
            sources.append(target)
        self.assertFalse(list(outside.glob('*.nano-file-*')))
        # Actual failed rename after a full staged write must clean the staging
        # path, while retaining the existing destination directory.
        destination = outside / 'destination';destination.mkdir()
        self.command(name + '-rename-refusal', [str(emitter),'--file-temporary','--entry-name','rename',str(wire),'-o',str(destination)],expected=1,cwd=outside)
        self.assertTrue(destination.is_dir())
        self.assertFalse(list(outside.glob('*.nano-file-*')))
        host = outside / 'host.c'
        host.write_text(r'''#include <nanolang/file/nanoisa/file_native_public.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
NvmFileRuntimeReport nvm_file_program_first(NvmFileHostGrant *,NvmFileScalar *);
NvmFileRuntimeReport nvm_file_program_second(NvmFileHostGrant *,NvmFileScalar *);
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"failure line %d\n",__LINE__);return 1;}}while(0)
static int descriptors(void){int n=0;for(int i=0;i<1024;i++)if(fcntl(i,F_GETFD)!=-1)n++;return n;}
int main(void){NvmFileHostGrant *g=NULL;NvmFileScalar out,old;memset(&out,0xa5,sizeof out);old=out;
NvmFileRuntimeReport r=nvm_file_program_first(NULL,&out);
REQUIRE(r.status==NVM_FILE_RUNTIME_INVALID && !r.acquired && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_host_grant_create_temporary_files(&g)==NVM_FILE_HOST_OK);
REQUIRE(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);
r=nvm_file_program_first(g,&out);REQUIRE(r.status==NVM_FILE_RUNTIME_BUSY && !memcmp(&out,&old,sizeof out));
r=nvm_file_program_second(g,&out);REQUIRE(r.status==NVM_FILE_RUNTIME_BUSY && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);nvm_file_host_leave();
int before=descriptors();r=nvm_file_program_first(g,&out);
REQUIRE(r.status==NVM_FILE_RUNTIME_OK && r.acquired && !r.cleanup.cleanup_failures && out.tag==TAG_INT && out.value==251);
r=nvm_file_program_second(g,&out);REQUIRE(r.status==NVM_FILE_RUNTIME_OK && out.value==251 && descriptors()==before);
REQUIRE(nvm_file_program_first(g,NULL).status==NVM_FILE_RUNTIME_INVALID);
REQUIRE(nvm_file_host_grant_revoke(g)==NVM_FILE_HOST_OK);out=old;
r=nvm_file_program_second(g,&out);REQUIRE(r.status==NVM_FILE_RUNTIME_STATE && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_host_grant_destroy(&g)==NVM_FILE_HOST_OK && !g);
puts("PASS outside-tree two-program C99 native package");return 0;}
''')
        flags = [f for f in self.flags if f not in ('-I.','-Isrc','-Isrc/nanoisa',
                 '-DNVM_FILE_VM_PRIVATE','-DNVM_FILE_NATIVE_PRIVATE','-DNVM_FILE_PUBLIC_ENGINE')]
        flags += ['-std=c99','-I' + str(self.installed / 'include')]
        archive = self.installed / 'lib/libnano_file_runtime.a'
        # I link the installed bridge independently of generated File programs.
        # Its required capture codec must be supplied by this archive alone.
        bridge_source = outside / 'installed-bridge.c'
        bridge_source.write_text('#include <nanolang/file/nanoisa/nvm_format.h>\n#include <nanolang/file/nanoisa/nvm_v2_sections.h>\nint main(void) {\n    NvmModule *source = nvm_module_new();\n    NvmModule *copy = NULL;\n    NvmV2Module view = {0};\n    if (!source) return 1;\n    if (nvm_v2_from_nvm_module(source, &view) != NVM_V2_OK) {\n        nvm_module_free(source); return 2;\n    }\n    NvmV2Result result = nvm_v2_to_nvm_module(&view, &copy);\n    nvm_v2_module_free(&view);\n    nvm_module_free(source);\n    if (result != NVM_V2_OK || !copy) return 3;\n    nvm_module_free(copy);\n    return 0;\n}\n')
        bridge_binary = outside / 'installed-bridge'
        self.command(name + '-outside-bridge-link', [*self.compiler,*flags,
            str(bridge_source),str(archive),*self.ldflags,'-o',str(bridge_binary)],cwd=outside)
        self.command(name + '-outside-bridge-run', [str(bridge_binary)],cwd=outside)
        for opt in ('-O0','-O2'):
            binary = outside / ('native-' + opt[1:])
            self.command(name + '-outside-' + opt[1:], [*self.compiler,*flags,opt,
                *map(str,sources),str(host),str(archive),*self.ldflags,'-o',str(binary)],cwd=outside)
            self.command(name + '-outside-run-' + opt[1:], [str(binary)],
                         'PASS outside-tree two-program C99 native package',cwd=outside)
            symbols = self.command(name + '-outside-symbols-' + opt[1:], ['nm',str(binary)],cwd=outside)
            for forbidden in ('nvm_file_vm_execute','file_vm_execute_serialized','nvm_file_execute_bytes','fvm_step','vm_execute'):
                self.assertNotIn(forbidden,symbols)
        for kind in ('abi','facts'):
            modified = outside / (kind + '.c')
            code = sources[0].read_text()
            if kind == 'abi':
                self.assertIn('nvm_file_runtime_native_abi(1u,',code)
                code = code.replace('nvm_file_runtime_native_abi(1u,','nvm_file_runtime_native_abi(2u,')
            else:
                code,count=re.subn(r'(s\.functions\)!=UINT64_C\()(\d+)',
                    lambda m:m[1]+str(int(m[2])+1),code,count=1)
                self.assertEqual(count,1)
            modified.write_text(code)
            refused_host=outside / (kind + '-host.c')
            refused_host.write_text(r'''#include <nanolang/file/nanoisa/file_public.h>
#include <string.h>
NvmFileRuntimeReport nvm_file_program_first(NvmFileHostGrant *,NvmFileScalar *);
NvmFileRuntimeReport nvm_file_program_second(NvmFileHostGrant *,NvmFileScalar *);
int main(void){NvmFileHostGrant *g=NULL;NvmFileScalar out,old;memset(&out,0xa5,sizeof out);old=out;
if(nvm_file_host_grant_create_temporary_files(&g)!=NVM_FILE_HOST_OK)return 1;
NvmFileRuntimeReport r=nvm_file_program_first(g,&out);
if(r.status!=NVM_FILE_RUNTIME_UNRESOLVED || r.acquired || memcmp(&out,&old,sizeof out))return 2;
r=nvm_file_program_second(g,&out);if(r.status!=NVM_FILE_RUNTIME_OK || out.value!=251)return 3;
return nvm_file_host_grant_destroy(&g)!=NVM_FILE_HOST_OK;}
''')
            for opt in ('-O0','-O2'):
                binary=outside / (kind + '-' + opt[1:])
                self.command(name + '-' + kind + '-build-' + opt[1:], [*self.compiler,*flags,opt,
                    str(modified),str(sources[1]),str(refused_host),str(archive),*self.ldflags,
                    '-o',str(binary)],cwd=outside)
                self.command(name + '-' + kind + '-run-' + opt[1:],[str(binary)],cwd=outside)
        self.command(name + '-duplicate-entry-refusal', [*self.compiler,*flags,
            str(sources[0]),str(sources[0]),str(sources[1]),str(host),str(archive),*self.ldflags,
            '-o',str(outside / 'duplicate')],expected=1,cwd=outside)
        duplicate_log=(self.artifacts / (name + '-duplicate-entry-refusal.log')).read_text()
        self.assertTrue('multiple definition' in duplicate_log or 'duplicate symbol' in duplicate_log)
        with wire.open('rb') as read_only_stdout:
            self.command(name + '-stdout-write-refusal', [str(emitter),'--file-temporary','--entry-name','stdout',str(wire)],
                         expected=1,cwd=outside,stdout_fd=read_only_stdout)
        (self.artifacts / (name + '-installed-artifacts.json')).write_text(json.dumps(
            {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(outside.iterdir())
             if p.is_file() and p != large},indent=2))

        # Retain exact installed bytes before invoking the actual uninstall.
        retained=self.artifacts / 'installed-before-uninstall'
        shutil.copytree(self.installed,retained)
        owned_headers=sorted((self.installed / 'include').rglob('*.h'))
        self.assertEqual(len(owned_headers),30)
        saved={str(p.relative_to(self.installed)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in self.installed.rglob('*') if p.is_file()}
        for rel,digest in saved.items():
            self.assertEqual(hashlib.sha256((retained / rel).read_bytes()).hexdigest(),digest)
        (self.artifacts / (name + '-pre-uninstall-artifacts.json')).write_text(json.dumps(saved,indent=2))
        (self.artifacts / (name + '-retained-install-sha256.json')).write_text(json.dumps(
            {str(retained / rel):digest for rel,digest in saved.items()},indent=2))
        sentinels=[self.installed / 'unrelated.txt',
                   self.installed / 'include/nanolang/file/unrelated.h',
                   self.installed / 'lib/unrelated.a',self.installed / 'bin/unrelated']
        for sentinel in sentinels:
            sentinel.write_bytes(b'I belong to another package.\n')
        self.command(name + '-actual-uninstall',['make','uninstall','PREFIX='+str(self.installed)])
        for path in [*owned_headers,archive,emitter]:
            self.assertFalse(path.exists(),str(path))
        for sentinel in sentinels:
            self.assertEqual(sentinel.read_bytes(),b'I belong to another package.\n')
        remaining={str(p.relative_to(self.installed)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in self.installed.rglob('*') if p.is_file()}
        (self.artifacts / (name + '-post-uninstall-artifacts.json')).write_text(json.dumps(remaining,indent=2))

    def test_cli_staging_faults(self):
        binary=self.artifacts / 'cli-faults'
        self.command('cli-faults-build', [*self.compiler,*self.flags,'-std=c99',
            'tests/nanoisa/test_file_public_cli.c','-o',str(binary)])
        self.command('cli-faults-run',[str(binary)],'PASS real CLI staging')

    def test_instrumented_native_corpus(self):
        self.qualify('instrumented', True)

    def test_linked_native_corpus_and_isolated_refusals(self):
        self.qualify('linked', False)


if __name__ == '__main__':
    unittest.main()
