"""I compare granted cyclic execution with the unchanged private corpus and package."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import unittest

from tests import test_file_cyclic_dispatch as private
from tests import test_file_public as acyclic

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*private.PROVIDERS, 'src/nanoisa/file_public_native.c',
             'src/nanovm/file_public_vm.c', 'src/nanoisa/file_host_grant.c',
             'src/nanoisa/file_cyclic_public_native.c', 'src/nanoisa/file_cyclic_public_abi.c',
             'src/nanovm/file_cyclic_public_vm.c']


class FileCyclicPublic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-cyclic-public-'))
        print(f'I retain public cyclic artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', '')),
                     '-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_PUBLIC_ENGINE',
                     '-DNVM_FILE_CYCLIC_VM_PRIVATE', '-DNVM_FILE_CYCLIC_NATIVE_PRIVATE',
                     '-g', '-Wall', '-Wextra', '-Werror', '-I.', '-Isrc', '-Isrc/nanoisa']
        if os.environ.get('NANO_FILE_RUNTIME_SANITIZERS', '1') != '0':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        stems = {Path(p).stem for p in PROVIDERS}
        cls.objects = list(dict.fromkeys(p for p in shlex.split(os.environ['FILE_RUNTIME_OBJECTS'])
                                        if Path(p).stem not in stems))
        cls.ldflags = shlex.split(os.environ.get('FILE_RUNTIME_LDFLAGS', '-lm -lcrypto -lffi'))
        cls.environment = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        for variable in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH'):
            cls.environment.pop(variable,None)
        cls.hooks = ['-include', 'tests/nanoisa/file_cyclic_dispatch_hooks.h',
                     '-Dnvm_file_runtime_begin=dispatch_begin',
                     '-Dnvm_file_runtime_frame_call=dispatch_call',
                     '-Dnvm_file_runtime_frame_return=dispatch_return',
                     '-Dnvm_file_runtime_service=dispatch_service',
                     '-Dnvm_file_runtime_cyclic_destroy=dispatch_destroy']
        executable = Path(sys.executable).resolve()
        (cls.artifacts / 'inputs.json').write_text(json.dumps({
            'compiler':cls.compiler,'flags':cls.flags,'providers':PROVIDERS,
            'ordinary_objects':cls.objects,'link_flags':cls.ldflags,
            'SDKROOT':os.environ.get('SDKROOT'),'LSAN_OPTIONS':'',
            'cleared_include_environment':['CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH'],
            'driver_python':str(executable),'driver_python_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
            'instrumentation':'listed rebuilt providers/generated C only; ordinary common objects separately inventoried'}, indent=2))

    # I reuse the bounded file-backed runner through its module, never importing
    # another TestCase into discovery. It records launch failures and group cleanup.
    stop_group = staticmethod(acyclic.FilePublic.stop_group)

    def command(self, *args, **kwargs):
        return acyclic.FilePublic.command(self, *args, **kwargs)

    def providers(self, name, instrument):
        objects = []
        alloc = ['-include','tests/nanoisa/file_runtime_hooks.h','-Dmalloc=file_test_malloc',
                 '-Dcalloc=file_test_calloc','-Drealloc=file_test_realloc','-Dfree=file_test_free']
        for source in PROVIDERS:
            stem = Path(source).stem
            if instrument and stem in ('file_runtime','file_vm_cyclic_private'):
                continue
            hooks = alloc if instrument and stem not in ('vm_ffi','file_host_grant') else []
            if stem == 'nsi_file':
                if not instrument:
                    hooks = ['-include','tests/nanoisa/file_runtime_hooks.h']
                hooks += ['-DFILE_RUNTIME_HOST_HOOKS']
            if stem == 'vm_ffi':
                hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                         '-Dffi_loader_open=file_runtime_loader_open','-Dfork=file_runtime_fork']
            if stem in ('file_vm_cyclic_private','file_cyclic_public_vm'):
                hooks += self.hooks
            actual = 'tests/nanoisa/file_cyclic_dispatch_values.c' if instrument and stem == 'nsi_file_values' else source
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler,*self.flags,'-O1',*hooks,
                '-c',actual,'-o',str(obj)])
            objects.append(str(obj))
        return objects

    def package(self, name):
        self.installed = self.artifacts / (name + '-installed')
        flags = [f for f in self.flags if not f.startswith('-DNVM_FILE_')]
        flags += ['-std=c99','-fPIC','-D_GNU_SOURCE']
        log = self.command(name + '-install', ['make','-B','-j2','install',
            'PREFIX='+str(self.installed),'CC='+shlex.join(self.compiler),
            'CFLAGS='+shlex.join(flags),'LDFLAGS='+shlex.join(self.ldflags)], timeout=900)
        lines = [s for s in log.splitlines() if ' -c src/nanoisa/file_host_grant.c ' in s]
        self.assertTrue(lines)
        self.assertTrue(all(s.rfind('-std=c11') > s.rfind('-std=c99') for s in lines))
        headers = sorted((self.installed / 'include').rglob('*.h'))
        self.assertEqual(len(headers),30)
        self.assertFalse(list(self.installed.rglob('*.inc')))
        self.assertFalse(list(self.installed.rglob('*.c')))
        self.assertFalse(any('private' in p.name for p in headers))
        self.archive = self.installed / 'lib/libnano_file_runtime.a'
        members = self.command(name + '-archive-members',['ar','t',str(self.archive)]).splitlines()
        for member in ('file_host_grant.o','file_cyclic_public_vm.o','file_cyclic_public_native.o','file_cyclic_public_abi.o'):
            self.assertEqual(members.count(member),1)
        for member in ('vm.o','file_vm_cyclic_private.o','nvm2c_file_cyclic_private.o'):
            self.assertNotIn(member,members)
        (self.artifacts / (name + '-installed-sha256.json')).write_text(json.dumps(
            {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.installed.rglob('*')) if p.is_file()},indent=2))

    def registry(self, name, rows, directory):
        lines = ['#include "src/nanoisa/file_cyclic_public.h"','#include <stdlib.h>','#include <string.h>',
                 'NvmFileHostGrant *file_cyclic_public_test_grant(void);']
        for i,status,size in rows:
            if status == 0:
                lines.append(f'extern NvmFileCyclicExecutionReport nf_case_{i}(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);')
            data = (directory / f'case-{i:03d}.nvm').read_bytes()
            self.assertEqual(len(data),size)
            lines.append(f'static const uint8_t wire_{i}[]={{'+','.join(map(str,data))+'};')
        first = next(i for i,status,_ in rows if status == 0)
        lines += ['void file_cyclic_public_native_busy(void){',
                  f'NvmFileCyclicExecutionReport r=nf_case_{first}((NvmFileHostGrant *)(uintptr_t)1,(const NvmFileCyclicOptions *)(uintptr_t)1,(NvmFileScalar *)(uintptr_t)1);',
                  'if(r.revision!=1 || r.runtime.status!=NVM_FILE_RUNTIME_BUSY || r.runtime.acquired || r.instructions_started || r.instruction_limit || r.fuel_exhausted || r.runtime.function!=UINT32_MAX || r.runtime.instruction!=UINT32_MAX)abort();}',
                  'NvmFileCyclicExecutionReport file_cyclic_registered(const uint8_t *bytes,size_t size,const NvmFileCyclicOptions *options,NvmFileRuntimeView *out){',
                  'NvmFileCyclicExecutionReport r={0};r.revision=1;r.instruction_limit=options?options->instruction_limit:0;r.runtime.function=r.runtime.instruction=UINT32_MAX;']
        for i,status,size in rows:
            lines.append(f'if(bytes && size=={size} && !memcmp(bytes,wire_{i},size)){{')
            if status == 0:
                lines += ['NvmFileScalar scalar,old;memset(&scalar,0xa5,sizeof scalar);old=scalar;',
                          f'r=nf_case_{i}(file_cyclic_public_test_grant(),options,out?&scalar:NULL);',
                          'if(r.runtime.status==NVM_FILE_RUNTIME_OK){NvmFileRuntimeView v={0};v.initialized=true;v.fields=1;v.type.tag=scalar.tag;',
                          'v.type.category=NVM_FILE_CATEGORY_UNKNOWN;v.type.global_index=v.type.catalog_ordinal=UINT32_MAX;v.values[0]=scalar.value;*out=v;}',
                          'else if(memcmp(&scalar,&old,sizeof scalar))abort();\nreturn r;}']
            else:
                lines += ['char *text=(char *)(uintptr_t)1;char error[256];',
                          'r.runtime.status=nvm2c_emit_file_cyclic_bytes(bytes,size,"refused",&text,error,sizeof error);',
                          f'if(r.runtime.status!={status} || text!=(char *)(uintptr_t)1)abort();\nreturn r;}}']
        lines += ['abort();}', '']
        path = self.artifacts / (name + '-registry.c')
        path.write_text('\n'.join(lines))
        return path

    def qualify(self, name, instrument):
        self.package(name)
        objects = self.providers(name,instrument)
        mode = ['-DHOSTED_INSTRUMENT'] if instrument else []
        outputs = {}
        for label,fixture in (('private','tests/nanoisa/test_file_cyclic_dispatch.c'),
                              ('public','tests/nanoisa/test_file_cyclic_public.c')):
            directory = self.artifacts / f'{name}-{label}-cases'
            directory.mkdir()
            capture = self.artifacts / f'{name}-{label}-capture'
            self.command(f'{name}-{label}-capture-build', [*self.compiler,*self.flags,'-O1',*mode,
                '-DFILE_CYCLIC_CAPTURE',fixture,'tests/nanoisa/test_file_cyclic_native_buffer.c',
                *objects,*self.objects,*self.ldflags,'-o',str(capture)])
            outputs[label] = self.command(f'{name}-{label}-capture-run',[str(capture),str(directory)])
        reference = [s for s in outputs['private'].splitlines() if s.startswith('TRACE ')]
        self.assertGreaterEqual(len(reference),28)
        self.assertEqual(reference,[s for s in outputs['public'].splitlines() if s.startswith('TRACE ')])
        directory = self.artifacts / f'{name}-public-cases'
        previous = self.artifacts / f'{name}-private-cases'
        rows = [tuple(map(int,s.split('\t'))) for s in (directory / 'cases.tsv').read_text().splitlines()]
        self.assertEqual((directory / 'cases.tsv').read_bytes(),(previous / 'cases.tsv').read_bytes())
        for i,_,_ in rows:
            self.assertEqual((directory / f'case-{i:03d}.nvm').read_bytes(),(previous / f'case-{i:03d}.nvm').read_bytes())
        registry = self.registry(name,rows,directory)
        generated = [(i,directory / f'case-{i:03d}.c') for i,status,_ in rows if status == 0]
        for _,source in generated:
            text = source.read_text()
            self.assertIn('#include <nanolang/file/nanoisa/file_cyclic_native_public.h>',text)
            self.assertIn('nvm_file_cyclic_program_case(',text)
            self.assertIn('static NvmFileCyclicExecutionReport nf_execute(',text)
            self.assertIn('nf_label_0:',text)
            self.assertNotIn('NVM_FILE_CYCLIC_NATIVE_PRIVATE',text)
            self.assertNotIn('src/',text)
            self.assertNotIn('fvm_step',text)
            self.assertNotIn('switch(',text[text.index('static NvmFileRuntimeStatus nf_function_'):])
        for opt in ('-O0','-O2'):
            compiled = []
            for i,source in generated:
                obj = self.artifacts / f'{name}-{opt[1:]}-{i}.o'
                self.command(f'{name}-{opt[1:]}-{i}-build', [*self.compiler,*self.flags,opt,'-std=c99',
                    '-I'+str(self.installed / 'include'),*self.hooks,
                    f'-Dnvm_file_cyclic_program_case=nf_case_{i}','-c',str(source),'-o',str(obj)])
                compiled.append(str(obj))
            binary = self.artifacts / f'{name}-{opt[1:]}-replay'
            self.command(f'{name}-{opt[1:]}-link', [*self.compiler,*self.flags,opt,*mode,
                'tests/nanoisa/test_file_cyclic_public.c',str(registry),*compiled,*objects,
                *self.objects,*self.ldflags,'-o',str(binary)])
            output = self.command(f'{name}-{opt[1:]}-run',[str(binary)],'PASS public cyclic native full corpus')
            self.assertEqual(reference,[s for s in output.splitlines() if s.startswith('TRACE ')])
        if not instrument:
            self.installed_controls(name,directory,generated)
        self.retain_and_uninstall(name)

    def outside_flags(self):
        flags=[]
        index=0
        while index<len(self.flags):
            flag=self.flags[index]
            index+=1
            if flag.startswith('-DNVM_FILE_'):
                continue
            if flag in ('-I','-iquote','-isystem'):
                self.assertLess(index,len(self.flags))
                path=self.flags[index]
                index+=1
                resolved=(ROOT/path).resolve()
                if resolved==ROOT or ROOT in resolved.parents:
                    continue
                flags += [flag,str(resolved)]
            elif flag.startswith('-I'):
                resolved=(ROOT/flag[2:]).resolve()
                if resolved==ROOT or ROOT in resolved.parents:
                    continue
                flags.append('-I'+str(resolved))
            else:
                self.assertFalse(flag.startswith('-include'),flag)
                flags.append(flag)
        return flags+['-std=c99','-I'+str(self.installed / 'include')]

    def installed_controls(self, name, directory, generated):
        outside = self.artifacts / (name + '-outside repository')
        outside.mkdir()
        vm = self.installed / 'bin/nano_vm'
        emitter = self.installed / 'bin/nvm2c'
        grant = ['--allow-temporary-files']
        cyclic = [*grant,'--file-cyclic','--file-instruction-limit']
        loop = directory / 'loop.nvm'
        for limit,expected in (('0',1),('31',1),('32',73),('1000000',73)):
            log = self.command(name + '-cli-loop-'+limit,[str(vm),*cyclic,limit,str(loop)],expected=expected,cwd=outside)
            if expected == 1:
                terminal = (self.artifacts / (name + '-cli-loop-'+limit+'.log')).read_text()
                self.assertIn('limit '+limit+', started '+limit+', exhausted 1',terminal)
        for limit,expected in (('0',1),('8',1),('9',11)):
            self.command(name + '-cli-initializer-'+limit,[str(vm),*cyclic,limit,str(directory / 'initializer.nvm')],expected=expected,cwd=outside)
        for label,expected in (('bool-false',0),('bool-true',1),('negative-int',255)):
            self.command(name + '-cli-'+label,[str(vm),*cyclic,'2',str(directory / (label+'.nvm'))],expected=expected,cwd=outside)
        invalid = ['', '-1','+1',' 1','1 ','1x','1000001','18446744073709551616']
        for i,limit in enumerate(invalid):
            self.command(name+f'-bad-limit-{i}',[str(vm),*cyclic,limit,str(loop)],expected=1,cwd=outside)
        combinations = [grant+['--file-cyclic'],grant+['--file-instruction-limit','32'],
            ['--file-cyclic','--file-instruction-limit','32'],
            cyclic+['32','--file-cyclic'],cyclic+['32','--file-instruction-limit','32']]
        combinations += [cyclic+['32',flag] for flag in ('--daemon','--verify-only','--check-shadows','--debug','--cop')]
        combinations += [cyclic+['32','--repeat','2'],cyclic+['32','--profile-isa','profile.json']]
        for i,args in enumerate(combinations):
            self.command(name+f'-bad-combination-{i}',[str(vm),*args,str(loop)],expected=1,cwd=outside)
        self.command(name+'-bad-guest',[str(vm),*cyclic,'32',str(loop),'--','guest'],expected=1,cwd=outside)
        self.command(name+'-old-cyclic-refusal',[str(vm),*grant,str(loop)],expected=1,cwd=outside)
        self.command(name+'-default-refusal',[str(vm),str(loop)],expected=1,cwd=outside)
        self.command(name+'-old-scalar-positive',[str(vm),*grant,str(directory/'negative-int.nvm')],expected=255,cwd=outside)
        sources = []
        for label,wire,use_cyclic in (('first',loop,True),('second',directory/'initializer.nvm',True),('old',directory/'negative-int.nvm',False),
                                      ('false',directory/'bool-false.nvm',True),('true',directory/'bool-true.nvm',True),
                                      ('negative',directory/'negative-int.nvm',True)):
            source = outside / (label+'.c')
            args = [str(emitter),'--file-temporary',*(['--file-cyclic'] if use_cyclic else []),'--entry-name',label,str(wire),'-o',str(source)]
            self.command(name+'-emit-'+label,args,cwd=outside)
            sources.append(source)
        destination=outside/'sentinel.c'
        bad_args=[['--file-cyclic'],['--file-temporary','--file-cyclic'],
            ['--file-temporary','--file-cyclic','--file-cyclic','--entry-name','bad'],
            ['--file-temporary','--file-cyclic','--entry-name','bad','--file-instruction-limit','32'],
            ['--file-temporary','--file-cyclic','--entry-name','bad-name']]
        for i,args in enumerate(bad_args):
            destination.write_bytes(b'previous output\n')
            self.command(name+f'-bad-emitter-{i}',[str(emitter),*args,str(loop),'-o',str(destination)],expected=1 if i==4 else 2,cwd=outside)
            self.assertEqual(destination.read_bytes(),b'previous output\n')
        # I exercise the unchanged I/O machinery through the newly selected route.
        emit_args=[str(emitter),'--file-temporary','--file-cyclic','--entry-name','io']
        bad_inputs=[outside/'missing.nvm',outside/'empty.nvm',outside/'truncated.nvm',outside/'large.nvm']
        bad_inputs[1].write_bytes(b'')
        bad_inputs[2].write_bytes(loop.read_bytes()[:-1])
        with bad_inputs[3].open('wb') as large:
            large.truncate(16*1024*1024+1)
        for i,path in enumerate(bad_inputs):
            destination.write_bytes(b'previous output\n')
            self.command(name+f'-bad-input-{i}',[*emit_args,str(path),'-o',str(destination)],expected=1,cwd=outside)
            self.assertEqual(destination.read_bytes(),b'previous output\n')
        output_directory=outside/'output-directory'
        output_directory.mkdir()
        self.command(name+'-bad-output-directory',[*emit_args,str(loop),'-o',str(output_directory)],expected=1,cwd=outside)
        self.assertTrue(output_directory.is_dir())
        self.assertFalse(list(outside.glob('*.nano-file-*')))
        with loop.open('rb') as read_only:
            self.command(name+'-bad-stdout',[*emit_args,str(loop)],expected=1,cwd=outside,stdout_fd=read_only)
        # Deterministic seek/read/close/write/rename faults remain the same shared CLI implementation.
        binary=self.artifacts/(name+'-cli-staging')
        self.command(name+'-cli-staging-build',[*self.compiler,*self.flags,'-std=c99',
            'tests/nanoisa/test_file_public_cli.c','-o',str(binary)])
        self.command(name+'-cli-staging-run',[str(binary)],'PASS real CLI staging')
        self.native_package(name,outside,sources,generated)
        # Compile only public host headers across each language form, without source includes.
        public_headers = ('file_cyclic_public.h','file_cyclic_report.h')
        for standard in ('c99','c11','c++11','c++17'):
            unit=outside/('header-'+standard+'.'+('cc' if '+' in standard else 'c'))
            unit.write_text('\n'.join('#include <nanolang/file/nanoisa/'+h+'>' for h in public_headers)+
                            '\nint main(void){NvmFileCyclicOptions o={1,0};return (int)o.instruction_limit;}\n')
            obj=outside/(unit.name+'.o')
            flags=[f for f in self.outside_flags() if not f.startswith('-std=')]
            self.command(name+'-header-'+standard,[*self.compiler,*flags,'-std='+standard,'-x',
                'c++' if '+' in standard else 'c','-c',str(unit),'-o',str(obj)],cwd=outside)

    def native_package(self, name, outside, sources, generated):
        host=outside/'host.c'
        host.write_text(r'''#include <nanolang/file/nanoisa/file_cyclic_native_public.h>
#include <stdio.h>
#include <string.h>
NvmFileCyclicExecutionReport nvm_file_cyclic_program_first(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
NvmFileCyclicExecutionReport nvm_file_cyclic_program_second(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
NvmFileRuntimeReport nvm_file_program_old(NvmFileHostGrant *,NvmFileScalar *);
NvmFileCyclicExecutionReport nvm_file_cyclic_program_false(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
NvmFileCyclicExecutionReport nvm_file_cyclic_program_true(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
NvmFileCyclicExecutionReport nvm_file_cyclic_program_negative(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"failure line %d\n",__LINE__);return 1;}}while(0)
int main(void){NvmFileHostGrant *g=NULL;NvmFileScalar out,old;memset(&out,0xa5,sizeof out);old=out;
NvmFileCyclicOptions options={1,32};
NvmFileCyclicExecutionReport absent=nvm_file_cyclic_program_first(NULL,&options,&out);
REQUIRE(absent.runtime.status==NVM_FILE_RUNTIME_INVALID && !absent.runtime.acquired && absent.instruction_limit==32 && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_host_grant_create_temporary_files(&g)==NVM_FILE_HOST_OK);
REQUIRE(nvm_file_cyclic_program_first(g,&options,NULL).runtime.status==NVM_FILE_RUNTIME_INVALID);
REQUIRE(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);
NvmFileCyclicExecutionReport r=nvm_file_cyclic_program_first((NvmFileHostGrant *)(uintptr_t)1,(const NvmFileCyclicOptions *)(uintptr_t)1,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_BUSY && !r.runtime.acquired && !r.instruction_limit && !r.instructions_started && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_program_old((NvmFileHostGrant *)(uintptr_t)1,(NvmFileScalar *)(uintptr_t)1).status==NVM_FILE_RUNTIME_BUSY);
REQUIRE(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);nvm_file_host_leave();
r=nvm_file_cyclic_program_first(g,&options,&out);REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==32 && out.value==73);
options.instruction_limit=1000000;r=nvm_file_cyclic_program_first(g,&options,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instruction_limit==1000000 && r.instructions_started==32 && out.value==73);
options.instruction_limit=31;out=old;r=nvm_file_cyclic_program_first(g,&options,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_LIMIT && r.fuel_exhausted && r.instructions_started==31 && !memcmp(&out,&old,sizeof out));
options.instruction_limit=9;r=nvm_file_cyclic_program_second(g,&options,&out);REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==9 && out.value==11);
REQUIRE(nvm_file_program_old(g,&out).status==NVM_FILE_RUNTIME_OK && out.value==-257);
options.instruction_limit=2;r=nvm_file_cyclic_program_false(g,&options,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==2 && out.tag==TAG_BOOL && out.value==0);
r=nvm_file_cyclic_program_true(g,&options,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==2 && out.tag==TAG_BOOL && out.value==1);
r=nvm_file_cyclic_program_negative(g,&options,&out);
REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==2 && out.tag==TAG_INT && out.value==-257);
options.instruction_limit=9;
REQUIRE(nvm_file_host_grant_revoke(g)==NVM_FILE_HOST_OK);out=old;
r=nvm_file_cyclic_program_first(g,&options,&out);REQUIRE(r.runtime.status==NVM_FILE_RUNTIME_STATE && r.instruction_limit==9 && !memcmp(&out,&old,sizeof out));
REQUIRE(nvm_file_host_grant_destroy(&g)==NVM_FILE_HOST_OK);puts("PASS installed cyclic and acyclic native units");return 0;}
''')
        for opt in ('-O0','-O2'):
            binary=outside/('native-'+opt[1:])
            self.command(name+'-native-link-'+opt[1:],[*self.compiler,*self.outside_flags(),opt,
                *map(str,sources),str(host),str(self.archive),*self.ldflags,'-o',str(binary)],cwd=outside)
            self.command(name+'-native-run-'+opt[1:],[str(binary)],'PASS installed cyclic and acyclic native units',cwd=outside)
            symbols=self.command(name+'-native-symbols-'+opt[1:],['nm',str(binary)],cwd=outside)
            for forbidden in ('nvm_file_vm_cyclic_execute','file_vm_cyclic_execute_serialized','nvm_file_execute_cyclic_bytes',
                              'nvm_file_execute_bytes','file_vm_execute_serialized','fvm_step','vm_execute','vm_core_execute'):
                self.assertNotIn(forbidden,symbols)
        self.command(name+'-duplicate-export',[*self.compiler,*self.outside_flags(),
            *map(str,[sources[0],*sources]),str(host),str(self.archive),*self.ldflags,'-o',str(outside/'duplicate')],expected=1,cwd=outside)
        log=(self.artifacts/(name+'-duplicate-export.log')).read_text()
        self.assertTrue('multiple definition' in log or 'duplicate symbol' in log)
        # Every actual compiled discrepancy refuses before acquisition and leaves output intact.
        plain=generated[0][1].read_text()
        alternatives=next(p.read_text() for _,p in generated if re.search(r'hosted_variant\(p,\d+,\d+,1,&variant\)',p.read_text()))
        references=next(p.read_text() for _,p in generated if 'reference.owner)!=UINT64_C(' in p.read_text())
        def alter(code,pattern):
            result,count=re.subn(pattern,lambda m:m[1]+str(int(m[2])+1),code,count=1)
            self.assertEqual(count,1)
            return result
        split=re.search(r'hosted_variant\(p,\d+,\d+,1,&variant\)',alternatives).end()
        controls=[('public-abi',plain.replace('cyclic_public_abi(1u,','cyclic_public_abi(2u,'),False),
                  ('public-size',plain.replace('sizeof copied,sizeof report,sizeof *out','sizeof copied+1,sizeof report,sizeof *out'),False),
                  ('native-abi',plain.replace('cyclic_native_abi(1u,','cyclic_native_abi(2u,'),False),
                  ('later-variant',alternatives[:split]+alter(alternatives[split:],r'(variant\.input\.owners\)!=UINT64_C\()(\d+)'),False),
                  ('reference',alter(references,r'(reference\.owner\)!=UINT64_C\()(\d+)'),False),
                  ('edge',alter(alternatives,r'(variant\.edge_variants\[0\]\)!=UINT64_C\()(\d+)'),False),
                  ('dead-label',plain.replace('\ngoto nf_label_0;\n','\ngoto nf_label_2;\n',1),True)]
        for label,code,acquired in controls:
            source=outside/(label+'.c');source.write_text(code)
            driver=outside/(label+'-host.c')
            driver.write_text('''#include <nanolang/file/nanoisa/file_cyclic_public.h>
#include <string.h>
NvmFileCyclicExecutionReport nvm_file_cyclic_program_case(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);
int main(void){NvmFileHostGrant *g=NULL;NvmFileScalar out,old;memset(&out,0xa5,sizeof out);old=out;
NvmFileCyclicOptions options={1,100000};if(nvm_file_host_grant_create_temporary_files(&g)!=NVM_FILE_HOST_OK)return 1;
NvmFileCyclicExecutionReport r=nvm_file_cyclic_program_case(g,&options,&out);
int ok=r.revision==1 && r.runtime.status=='''+('NVM_FILE_RUNTIME_STATE' if acquired else 'NVM_FILE_RUNTIME_UNRESOLVED')+
                ' && r.runtime.acquired=='+str(int(acquired))+''' && r.instruction_limit==100000 && !r.instructions_started && !r.fuel_exhausted && !memcmp(&out,&old,sizeof out);
if(nvm_file_host_grant_destroy(&g)!=NVM_FILE_HOST_OK)return 2;
return !ok;}
''')
            for opt in ('-O0','-O2'):
                binary=outside/(label+'-'+opt[1:])
                self.command(name+'-'+label+'-build-'+opt[1:],[*self.compiler,*self.outside_flags(),opt,
                    str(source),str(driver),str(self.archive),*self.ldflags,'-o',str(binary)],cwd=outside)
                self.command(name+'-'+label+'-run-'+opt[1:],[str(binary)],cwd=outside)

    def retain_and_uninstall(self,name):
        retained=self.artifacts/(name+'-installed-before-uninstall')
        shutil.copytree(self.installed,retained)
        files={str(p.relative_to(self.installed)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in self.installed.rglob('*') if p.is_file()}
        for rel,digest in files.items():
            self.assertEqual(hashlib.sha256((retained/rel).read_bytes()).hexdigest(),digest)
        (self.artifacts/(name+'-retained-install.json')).write_text(json.dumps(
            {str(retained/rel):digest for rel,digest in files.items()},indent=2))
        sentinel=self.installed/'include/nanolang/file/unrelated.h'
        sentinel.write_bytes(b'I belong to another package.\n')
        self.command(name+'-uninstall',['make','uninstall','PREFIX='+str(self.installed)])
        self.assertEqual(sentinel.read_bytes(),b'I belong to another package.\n')
        for rel in files:
            self.assertFalse((self.installed/rel).exists(),rel)

    def test_instrumented_public_cyclic_corpus(self):
        self.qualify('instrumented',True)

    def test_linked_public_cyclic_and_installed_package(self):
        self.qualify('linked',False)


if __name__ == '__main__':
    unittest.main()
