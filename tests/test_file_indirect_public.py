"""I compare granted indirect execution with the unchanged private corpus and package."""
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

from tests import test_file_indirect_dispatch as private
from tests import test_file_public as acyclic

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = [*private.PROVIDERS, 'src/nanoisa/file_public_native.c',
             'src/nanovm/file_public_vm.c', 'src/nanoisa/file_host_grant.c',
             'src/nanoisa/file_indirect_public_native.c', 'src/nanoisa/file_indirect_public_abi.c',
             'src/nanovm/file_indirect_public_vm.c', 'src/nanoisa/file_cyclic_public_native.c',
             'src/nanovm/file_cyclic_public_vm.c']


class FileIndirectPublic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-file-indirect-public-'))
        print(f'I retain public indirect artifacts at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('NANO_FILE_RUNTIME_CC', 'cc'))
        cls.flags = [*shlex.split(os.environ.get('NANO_FILE_RUNTIME_CFLAGS', '')),
                     '-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_PUBLIC_ENGINE',
                     '-DNVM_FILE_INDIRECT_VM_PRIVATE', '-DNVM_FILE_INDIRECT_NATIVE_PRIVATE',
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
        cls.hooks = ['-include', 'tests/nanoisa/file_indirect_dispatch_hooks.h',
                     '-Dnvm_file_runtime_begin=dispatch_begin',
                     '-Dnvm_file_runtime_frame_call=dispatch_call',
                     '-Dnvm_file_runtime_frame_return=dispatch_return',
                     '-Dnvm_file_runtime_service=dispatch_service',
                     '-Dnvm_file_runtime_indirect_destroy=dispatch_destroy']
        executable = Path(sys.executable).resolve()
        (cls.artifacts / 'inputs.json').write_text(json.dumps({
            'compiler':cls.compiler,'flags':cls.flags,'providers':PROVIDERS,
            'ordinary_objects':cls.objects,'link_flags':cls.ldflags,
            'SDKROOT':os.environ.get('SDKROOT'),'LSAN_OPTIONS':'',
            'cleared_include_environment':['CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH'],
            'driver_python':str(executable),'driver_python_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
            'instrumentation':'allocation hooks cover listed rebuilt providers/generated C; common objects and the installed archive retain their Make build flags, recorded separately per phase'}, indent=2))

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
            if instrument and stem in ('file_runtime','file_vm_indirect_private'):
                continue
            hooks = alloc if instrument and stem not in ('vm_ffi','file_host_grant') else []
            if stem == 'nsi_file':
                if not instrument:
                    hooks = ['-include','tests/nanoisa/file_runtime_hooks.h']
                hooks += ['-DFILE_RUNTIME_HOST_HOOKS']
            if stem == 'vm_ffi':
                hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                         '-Dffi_loader_open=file_runtime_loader_open','-Dfork=file_runtime_fork']
            if stem in ('file_vm_indirect_private','file_indirect_public_vm'):
                hooks += self.hooks
            actual = 'tests/nanoisa/file_indirect_dispatch_values.c' if instrument and stem == 'nsi_file_values' else source
            obj = self.artifacts / f'{name}-{stem}.o'
            self.command(f'{name}-{stem}-build', [*self.compiler,*self.flags,'-O1',*hooks,
                '-c',actual,'-o',str(obj)])
            objects.append(str(obj))
        return objects

    def package(self, name):
        self.installed = self.artifacts / (name + '-installed')
        self.command(name + '-install', ['make','-f','Makefile.gnu','install-file-public-runtime',
            'PREFIX='+str(self.installed)], timeout=300)
        self.archive = self.installed / 'lib/libnano_file_runtime.a'
        headers = sorted((self.installed / 'include').rglob('*.h'))
        self.assertEqual(len(headers),39)
        self.assertFalse(list(self.installed.rglob('*.inc')))
        self.assertFalse(list(self.installed.rglob('*.c')))
        self.assertFalse(any('private' in p.name for p in headers))
        members = self.command(name+'-members',['ar','t',str(self.archive)]).splitlines()
        for member in ('file_indirect_public_vm.o','file_indirect_public_native.o','file_indirect_public_abi.o'):
            self.assertEqual(members.count(member),1)
        binaries=self.installed/'bin';binaries.mkdir()
        for tool in ('nano_vm','nvm2c'):
            shutil.copy2(ROOT/'bin'/tool,binaries/tool)
        for member in ('vm.o','file_vm_indirect_private.o','nvm2c_file_indirect_private.o'):
            self.assertNotIn(member,members)

    def registry(self, name, rows, directory):
        lines = ['#include "src/nanoisa/file_indirect_public.h"','#include <stdlib.h>','#include <string.h>',
                 'NvmFileHostGrant *file_indirect_public_test_grant(void);']
        for i,status,size in rows:
            if status == 0:
                lines.append(f'extern NvmFileIndirectExecutionReport nf_case_{i}(NvmFileHostGrant *,const NvmFileIndirectOptions *,NvmFileScalar *);')
            data = (directory / f'case-{i:03d}.nvm').read_bytes()
            self.assertEqual(len(data),size)
            lines.append(f'static const uint8_t wire_{i}[]={{'+','.join(map(str,data))+'};')
        first = next(i for i,status,_ in rows if status == 0)
        lines += ['void file_indirect_public_native_busy(void){',
                  f'NvmFileIndirectExecutionReport r=nf_case_{first}((NvmFileHostGrant *)(uintptr_t)1,(const NvmFileIndirectOptions *)(uintptr_t)1,(NvmFileScalar *)(uintptr_t)1);',
                  'if(r.revision!=1 || r.runtime.status!=NVM_FILE_RUNTIME_BUSY || r.runtime.acquired || r.instructions_started || r.instruction_limit || r.fuel_exhausted || r.runtime.function!=UINT32_MAX || r.runtime.instruction!=UINT32_MAX)abort();}',
                  'NvmFileIndirectExecutionReport file_indirect_registered(const uint8_t *bytes,size_t size,const NvmFileIndirectOptions *options,NvmFileRuntimeView *out){',
                  'NvmFileIndirectExecutionReport r={0};r.revision=1;r.instruction_limit=options?options->instruction_limit:0;r.runtime.function=r.runtime.instruction=UINT32_MAX;']
        for i,status,size in rows:
            lines.append(f'if(bytes && size=={size} && !memcmp(bytes,wire_{i},size)){{')
            if status == 0:
                lines += ['NvmFileScalar scalar,old;memset(&scalar,0xa5,sizeof scalar);old=scalar;',
                          f'r=nf_case_{i}(file_indirect_public_test_grant(),options,out?&scalar:NULL);',
                          'if(r.runtime.status==NVM_FILE_RUNTIME_OK){NvmFileRuntimeView v={0};v.initialized=true;v.fields=1;v.type.tag=scalar.tag;',
                          'v.type.category=NVM_FILE_CATEGORY_UNKNOWN;v.type.global_index=v.type.catalog_ordinal=UINT32_MAX;v.values[0]=scalar.value;*out=v;}',
                          'else if(memcmp(&scalar,&old,sizeof scalar))abort();\nreturn r;}']
            else:
                lines += ['char *text=(char *)(uintptr_t)1;char error[256];',
                          'r.runtime.status=nvm2c_emit_file_indirect_bytes(bytes,size,"refused",&text,error,sizeof error);',
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
        for label,fixture in (('private','tests/nanoisa/test_file_indirect_dispatch.c'),
                              ('public','tests/nanoisa/test_file_indirect_public.c')):
            directory = self.artifacts / f'{name}-{label}-cases'
            directory.mkdir()
            capture = self.artifacts / f'{name}-{label}-capture'
            self.command(f'{name}-{label}-capture-build', [*self.compiler,*self.flags,'-O1',*mode,
                '-DFILE_INDIRECT_CAPTURE',fixture,'tests/nanoisa/test_file_indirect_native_buffer.c',
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
            self.assertIn('#include <nanolang/file/nanoisa/file_indirect_native_public.h>',text)
            self.assertIn('nvm_file_indirect_program_case(',text)
            self.assertIn('static NvmFileIndirectExecutionReport nf_execute(',text)
            self.assertIn('nf_label_0:',text)
            self.assertNotIn('NVM_FILE_INDIRECT_NATIVE_PRIVATE',text)
            self.assertNotIn('src/',text)
            self.assertNotIn('fvm_step',text)
            self.assertIn('nvm_file_runtime_indirect_plan',text)
        for opt in ('-O0','-O2'):
            compiled = []
            for i,source in generated:
                obj = self.artifacts / f'{name}-{opt[1:]}-{i}.o'
                self.command(f'{name}-{opt[1:]}-{i}-build', [*self.compiler,*self.flags,opt,'-std=c99',
                    '-I'+str(self.installed / 'include'),*self.hooks,
                    f'-Dnvm_file_indirect_program_case=nf_case_{i}','-c',str(source),'-o',str(obj)])
                compiled.append(str(obj))
            binary = self.artifacts / f'{name}-{opt[1:]}-replay'
            self.command(f'{name}-{opt[1:]}-link', [*self.compiler,*self.flags,opt,*mode,
                'tests/nanoisa/test_file_indirect_public.c',str(registry),*compiled,*objects,
                *self.objects,*self.ldflags,'-o',str(binary)])
            output = self.command(f'{name}-{opt[1:]}-run',[str(binary)],'PASS public indirect native full corpus')
            self.assertEqual(reference,[s for s in output.splitlines() if s.startswith('TRACE ')])
        if not instrument:
            self.installed_controls(name,directory,generated)


    def installed_controls(self, name, directory, generated):
        from tests import test_file_cyclic_public as cyclic
        flags = cyclic.FileCyclicPublic.outside_flags(self)
        outside = self.artifacts / (name+'-outside repository')
        outside.mkdir()
        for standard in ('c99','c11','c++11','c++17'):
            unit=outside/('header-'+standard+'.'+('cc' if '+' in standard else 'c'))
            unit.write_text('#include <nanolang/file/nanoisa/file_indirect_public.h>\nint main(void){NvmFileIndirectOptions o={1,0};return (int)o.instruction_limit;}\n')
            self.command(name+'-header-'+standard,[*self.compiler,*[f for f in flags if not f.startswith('-std=')],
                '-std='+standard,'-x','c++' if '+' in standard else 'c','-c',str(unit),'-o',str(unit)+'.o'],cwd=outside)
        # I select the actual 256-shared-formal module by byte identity.
        wire=(directory/'borrowed-maximum.nvm').read_bytes()
        selected=next(p for i,p in generated if (directory / ('case-%03d.nvm'%i)).read_bytes()==wire)
        source=outside/'program.c';source.write_bytes(selected.read_bytes())
        cli=self.installed/'bin/nano_vm';emitter=self.installed/'bin/nvm2c'
        grant=['--allow-temporary-files'];profile=[*grant,'--file-indirect','--file-instruction-limit']
        maximum=directory/'borrowed-maximum.nvm'
        self.command(name+'-cli-maximum',[str(cli),*profile,'1000000',str(maximum)],expected=42,cwd=outside)
        self.command(name+'-cli-zero',[str(cli),*profile,'0',str(maximum)],expected=1,cwd=outside)
        for label,extra in [('default',[]),('old',grant),('cyclic',grant+['--file-cyclic','--file-instruction-limit','1000000'])]:
            self.command(name+'-cli-refused-'+label,[str(cli),*extra,str(maximum)],expected=1,cwd=outside)
        invalid=[profile+[limit] for limit in ('','-1','+1',' 1','1 ','1x','1000001','18446744073709551616')]
        invalid += [grant+['--file-indirect'],['--file-indirect','--file-instruction-limit','42'],
                    profile+['42','--file-indirect'],profile+['42','--file-cyclic'],
                    grant+['--file-cyclic','--file-indirect','--file-instruction-limit','42']]
        invalid += [profile+['42',flag] for flag in ('--daemon','--verify-only','--check-shadows','--debug','--cop')]
        for index,args in enumerate(invalid):
            self.command(name+'-cli-invalid-'+str(index),[str(cli),*args,str(maximum)],expected=1,cwd=outside)
        self.command(name+'-cli-emit',[str(emitter),'--file-temporary','--file-indirect','--entry-name','case',str(maximum),'-o',str(source)],cwd=outside)
        sentinel=outside/'sentinel.c'
        bad_emit=[['--file-indirect'],['--file-temporary','--file-indirect'],
                  ['--file-temporary','--file-indirect','--file-cyclic','--entry-name','case'],
                  ['--file-temporary','--file-indirect','--file-indirect','--entry-name','case'],
                  ['--file-temporary','--file-indirect','--entry-name','bad-name']]
        for index,args in enumerate(bad_emit):
            sentinel.write_bytes(b'previous output\n')
            self.command(name+'-emitter-invalid-'+str(index),[str(emitter),*args,str(maximum),'-o',str(sentinel)],expected=1 if index==4 else 2,cwd=outside)
            self.assertEqual(sentinel.read_bytes(),b'previous output\n')
        common = r'''#include <nanolang/file/nanoisa/file_indirect_public.h>
#include <stdio.h>
#include <string.h>
NvmFileIndirectExecutionReport nvm_file_indirect_program_case(NvmFileHostGrant *,const NvmFileIndirectOptions *,NvmFileScalar *);
static const uint8_t bytes[]={'''+','.join(map(str,wire))+r'''};
int main(void){NvmFileHostGrant *g=NULL;NvmFileScalar out,old;memset(&out,0xa5,sizeof out);old=out;
NvmFileIndirectOptions options={1,1000000};NvmFileIndirectExecutionReport r;
if(nvm_file_host_grant_create_temporary_files(&g)!=NVM_FILE_HOST_OK)return 1;
(void)bytes;
#ifdef RUN_VM
#define RUN() nvm_file_execute_indirect_bytes(g,bytes,sizeof bytes,&options,&out)
#else
#define RUN() nvm_file_indirect_program_case(g,&options,&out)
#endif
r=RUN();if(r.runtime.status!=NVM_FILE_RUNTIME_OK)return 2;
printf("%u %u %lld %llu %u\n",r.runtime.status,out.tag,(long long)out.value,(unsigned long long)r.instructions_started,r.fuel_exhausted);
options.instruction_limit=0;out=old;r=RUN();
if(r.runtime.status!=NVM_FILE_RUNTIME_LIMIT || !r.fuel_exhausted || r.instructions_started || memcmp(&out,&old,sizeof out))return 3;
if(nvm_file_host_grant_revoke(g)!=NVM_FILE_HOST_OK)return 4;
r=RUN();if(r.runtime.status!=NVM_FILE_RUNTIME_STATE || r.runtime.acquired || memcmp(&out,&old,sizeof out))return 5;
return nvm_file_host_grant_destroy(&g)!=NVM_FILE_HOST_OK;}
'''
        host=outside/'host.c';host.write_text(common)
        vm=outside/'vm'
        self.command(name+'-installed-vm-build',[*self.compiler,*flags,'-DRUN_VM',str(host),str(self.archive),*self.ldflags,'-o',str(vm)],cwd=outside)
        expected=self.command(name+'-installed-vm-run',[str(vm)],cwd=outside)
        for opt in ('-O0','-O2'):
            binary=outside/('native-'+opt[1:])
            self.command(name+'-installed-'+opt[1:]+'-build',[*self.compiler,*flags,opt,str(source),str(host),str(self.archive),*self.ldflags,'-o',str(binary)],cwd=outside)
            self.assertEqual(expected,self.command(name+'-installed-'+opt[1:]+'-run',[str(binary)],cwd=outside))
            symbols=self.command(name+'-installed-'+opt[1:]+'-symbols',['nm',str(binary)],cwd=outside)
            for forbidden in ('nvm_file_execute_indirect_bytes','file_vm_indirect_execute_serialized','fvm_step','vm_execute','file_indirect_native_emit_serialized'):
                self.assertNotIn(forbidden,symbols)

        original=source.read_text()
        controls=[('abi',original.replace('indirect_public_abi(1u,','indirect_public_abi(2u,')),
                  ('size',original.replace('sizeof copied,sizeof report,sizeof *out','sizeof copied+1,sizeof report,sizeof *out'))]
        altered,count=re.subn(r'(in\.call_references\[0\]!=)(\d+)',lambda m:m[1]+str(int(m[2])+1),original,count=1)
        self.assertEqual(count,1);controls.append(('reference-map',altered))
        failure_host=outside/'failure-host.c'
        failure_host.write_text(common[:common.index('r=RUN();')]+'''r=RUN();
int ok=r.runtime.status==NVM_FILE_RUNTIME_UNRESOLVED && !r.runtime.acquired && !r.instructions_started && !memcmp(&out,&old,sizeof out);
if(nvm_file_host_grant_destroy(&g)!=NVM_FILE_HOST_OK)return 6;
return !ok;}''')
        for label,code in controls:
            self.assertNotEqual(original,code)
            tamper=outside/(label+'.c');tamper.write_text(code)
            binary=outside/label
            self.command(name+'-tamper-'+label+'-build',[*self.compiler,*flags,'-O2',str(tamper),str(failure_host),str(self.archive),*self.ldflags,'-o',str(binary)],cwd=outside)
            self.command(name+'-tamper-'+label+'-run',[str(binary)],cwd=outside)

    def test_instrumented_public_indirect_corpus(self):
        self.qualify('instrumented',True)

    def test_linked_public_indirect_and_installed_package(self):
        self.qualify('linked',False)


if __name__ == '__main__':
    unittest.main()
