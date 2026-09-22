"""I qualify actual installed compilers outside the source checkout.

My orchestrator supplies fresh bootstrap/providers. Common installed products
remain ordinary; only native_sdk_probe plus module_build_dir use selected
sanitizer instrumentation. I retain every command and exact package inventory.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import sys
import tempfile
import unittest
from tests.native_sdk_runner import run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import native_sdk as sdk
sys.path.pop(0)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):value.update(data)
    return value.hexdigest()


def replace(path, data, mode=None):
    path=Path(path);temp=path.with_name(path.name+'.sdk-replacement')
    with temp.open('xb') as stream:stream.write(data)
    temp.chmod(mode if mode is not None else stat.S_IMODE(path.stat().st_mode))
    os.replace(temp,path)


def spans(data):
    return {parts[0].decode():os.fsdecode(bytes.fromhex(parts[1].decode()))
            for line in data.splitlines() if (parts:=line.split()) and
            parts[0] in (b'ROOT',b'FIRST',b'SECOND',b'CACHE') and len(parts)==2}


class NativeSdk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-installed-sdk-',dir=os.environ.get('NANO_SDK_REPORT_DIR'))).resolve()
        cls.outside=cls.work/'outside';cls.outside.mkdir()
        cls.prefix=cls.work/'prefix with spaces'
        cls.private=cls.work/'private';cls.private.mkdir()
        cls.cc=shlex.split(os.environ.get('NANO_SDK_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_SDK_CFLAGS',''))
        cls.links=shlex.split(os.environ.get('NANO_SDK_LDFLAGS','-lcrypto'))
        cls.env=dict(NANOLANG_SDK_ROOT=None,NANO_BUILD_CACHE=None,NANO_MODULE_PATH=None,
                     NANO_VIRT_LIB=None,TMPDIR=str(cls.private),NANO_SHADOW_TRACE='1',
                     CC=shlex.join(cls.cc),NANO_CC=shlex.join(cls.cc))
        print('I retain installed SDK artifacts at',cls.work,flush=True)
        cls.command('inventory-check',[sys.executable,ROOT/'scripts/generate_native_sdk_inventory.py','--check'])
        if os.environ.get('NANO_SDK_EXPECT_CLEAN')=='1':
            existing=[str(p.relative_to(ROOT)) for name in ('bin','obj','lib')
                      for p in (ROOT/name).rglob('*') if p.is_file() and p.name!='.gitkeep']
            (cls.work/'clean-install-inputs.json').write_text(json.dumps(existing)+'\n')
            if existing:raise AssertionError(('I require a clean actual install build',existing))
        cls.command('actual-make-install',['make','-j2','CC='+shlex.join(cls.cc),
                    'PREFIX='+str(cls.prefix),'install'],cwd=ROOT,timeout=3600)
        cls.generation=(cls.prefix/'bin/nanoc').resolve().parents[1]
        identity,cls.rows=sdk.verify(cls.generation,cls.generation.name)
        cls.baseline={r['path']:r for r in cls.rows}
        (cls.work/'installed-inventory.json').write_text(json.dumps(dict(root=str(cls.generation),identity=identity,rows=cls.rows),indent=2)+'\n')
        cls.probe=cls.work/'native-sdk-probe'
        probe_flags=['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',ROOT/'src']
        if os.environ.get('NANO_SDK_SANITIZERS')=='1':probe_flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.command('probe-build',[*cls.cc,*cls.flags,*probe_flags,ROOT/'tests/native_sdk_probe.c',ROOT/'src/runtime/module_build_dir.c',*cls.links,'-o',cls.probe])
        cls.observer=cls.work/'compiler-observer.py'
        cls.observer.write_text('#!'+sys.executable+'\nimport json,os,subprocess,sys\n'
            'fd=os.open(os.environ["SDK_CC_LOG"],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\n'
            'os.write(fd,(json.dumps(sys.argv[1:])+"\\n").encode());os.close(fd)\n'
            'sys.exit(subprocess.call(json.loads(os.environ["SDK_REAL_CC"])+sys.argv[1:]))\n')
        cls.observer.chmod(0o755)
        cls.command('actual-make-reinstall',['make','-j2','CC='+shlex.join(cls.cc),'PREFIX='+str(cls.prefix),'install'],cwd=ROOT,timeout=3600)
        assert (cls.prefix/'bin/nanoc').resolve().parents[1]==cls.generation

    @classmethod
    def command(cls,name,args,cwd=None,extra=None,expected=(0,),timeout=180):
        env=dict(cls.env);env.update(extra or {})
        return run(cls.work,name,args,cwd or cls.outside,env,expected,timeout)

    def assert_package_unchanged(self):
        identity,rows=sdk.verify(self.generation,self.generation.name)
        self.assertEqual(rows,self.rows);self.assertEqual(identity,self.generation.name)

    @contextmanager
    def hidden_source(self):
        hidden=ROOT.with_name(ROOT.name+'-sdk-hidden')
        self.assertFalse(hidden.exists())
        ROOT.rename(hidden)
        try:
            self.assertFalse(ROOT.exists())
            yield
        finally:
            hidden.rename(ROOT)

    def fault_generation(self,name,change):
        parent=self.work/('fault-'+name);parent.mkdir()
        clone=parent/'stage'
        shutil.copytree(self.generation,clone,copy_function=os.link)
        change(clone)
        rows=[]
        for old in self.rows:
            p=clone/old['path']
            if p.exists():rows.append(sdk.row(old['path'],p))
        identity,data=sdk.manifest(rows)
        replace(clone/sdk.MANIFEST,data)
        target=parent/identity;clone.rename(target)
        (parent/'case.json').write_text(json.dumps(dict(identity=identity,rows=rows),indent=2)+'\n')
        return target

    def test_cop_exec_installed_source_hidden(self):
        # I use the actual installed generation and existing hiding boundary.
        self.command('cop-exec-fixtures', ['make', '-j2', 'CC='+shlex.join(self.cc),
                     'cop-exec-fixtures'], cwd=ROOT, timeout=1200)
        control=self.work/'cop-exec-control';provider=self.work/'exec-provider.so'
        shutil.copy2(ROOT/'obj/test_cop_exec', control)
        shutil.copy2(ROOT/'obj/exec_provider.so', provider)
        before={str(p):digest(p) for p in (control,provider,self.generation/'bin/nano_cop')}
        assembly=self.work/'cop-discovery.nasm';module=self.work/'cop-discovery.nvm'
        assembly.write_text('.entry main\n.import "" "abs" int int\n'
            '.function main 0 0 0 int 1\nPUSH_I64 -17\nCALL_EXTERN 0\n'
            'PUSH_I64 17\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        self.command('cop-installed-assemble',[self.prefix/'bin/nanoisa','asm',assembly,'-o',module])
        with self.hidden_source():
            # No override: the installed CLI must discover its own SDK worker.
            self.command('cop-installed-cli',[self.prefix/'bin/nano_vm','--isolate-ffi',module])
            output,_,_=self.command('cop-installed-hidden', [control,provider],
                extra={'NANOLANG_SDK_ROOT':self.generation}, timeout=180)
            self.assertEqual(output.count(b'exit 0 before parent cleanup.'),6)
            self.assertIn(b'I checked exec startup, independent images, transports, descriptors and concurrent workers.',output)
        self.assertEqual(before,{str(p):digest(p) for p in (control,provider,self.generation/'bin/nano_cop')})
        self.assert_package_unchanged()

    def test_0_nested_session_timeout_cleanup(self):
        directory=self.work/'nested-supervision';directory.mkdir()
        marker=directory/'child.json';survived=directory/'survived'
        script=directory/'nested.py'
        script.write_text('import json,os,subprocess,sys,time\nfrom pathlib import Path\n'
            'if len(sys.argv)>3:\n'
            ' Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"group":os.getpgrp()})+"\\n")\n'
            ' time.sleep(6);Path(sys.argv[2]).write_text("escaped");time.sleep(30)\n'
            'else:\n'
            ' subprocess.Popen([sys.executable,__file__,sys.argv[1],sys.argv[2],"child"],start_new_session=True)\n'
            ' time.sleep(30)\n')
        with self.assertRaises(AssertionError):
            run(directory,'nested-timeout',[sys.executable,script,marker,survived],self.outside,
                self.env,timeout=2,track_descendants=True)
        status=json.loads((directory/'nested-timeout-status.json').read_text())
        self.assertTrue(status['timeout']);self.assertEqual(status['returncode'],124)
        self.assertTrue(status['group_absent']);self.assertTrue(status['descendants_absent'])
        self.assertEqual(status['cleanup_errors'],[])
        child=json.loads(marker.read_text());inventory=json.loads((directory/'nested-timeout-descendants.json').read_text())
        self.assertIn(str(child['pid']),inventory['owned']);self.assertEqual(inventory['remaining'],[])
        self.assertTrue(any(row['group']==child['group'] for row in inventory['cleanup']))
        with self.assertRaises(ProcessLookupError):os.kill(child['pid'],0)
        self.assertFalse(survived.exists())

    def test_a_root_identity_private_products_and_lists(self):
        out,_,_=self.command('installed-root',[self.probe,'root','0','installed'],extra={'NANOLANG_SDK_ROOT':self.generation})
        self.assertEqual(spans(out)['ROOT'],str(self.generation))
        callback_out,_,_=self.command('loader-callback',[self.probe,'callback','0','installed'],extra={'NANOLANG_SDK_ROOT':self.generation})
        self.assertEqual(callback_out.count(b'CALLBACK before private cleanup'),1)
        self.assertIn(b'PASS callback and private cleanup order',callback_out)
        self.command('short-root',[self.probe,'short','3','installed'],extra={'NANOLANG_SDK_ROOT':self.generation})
        out,_,_=self.command('private-objects',[self.probe,'objects','0','installed'],extra={'NANOLANG_SDK_ROOT':self.generation})
        values=spans(out)
        for name in ('FIRST','SECOND','CACHE'):
            self.assertTrue(Path(values[name]).is_relative_to(self.private));self.assertFalse(Path(values[name]).exists())
        explicit=self.work/'user-cache';explicit.mkdir();(explicit/'sentinel').write_text('owned by user')
        out,_,_=self.command('explicit-cache',[self.probe,'objects','0','installed'],extra={'NANOLANG_SDK_ROOT':self.generation,'NANO_BUILD_CACHE':explicit})
        self.assertEqual(spans(out)['CACHE'],str(explicit));self.assertEqual((explicit/'sentinel').read_text(),'owned by user')
        out,_,_=self.command('private-lists',[self.probe,'lists','0','installed'],extra={'NANOLANG_SDK_ROOT':self.generation})
        generated=Path(spans(out)['FIRST'])
        self.assertTrue((generated/'list_SdkPair.h').is_file());self.assertTrue((generated/'list_SdkPair.c').is_file())
        source=generated/'main.c'
        source.write_text('#include <stdint.h>\n#include <assert.h>\ntypedef struct { int64_t value; } SdkPair;\n'
            '#include "list_SdkPair.c"\nint main(void) { List_SdkPair *p=nl_list_SdkPair_new(); '
            'SdkPair value={37}; nl_list_SdkPair_push(p,value); assert(nl_list_SdkPair_get(p,0).value==37); '
            'nl_list_SdkPair_free(p); return 0; }\n')
        exe=generated/'list-program'
        self.command('generated-list-build',[*self.cc,*self.flags,'-std=c99','-Wall','-Wextra','-Werror',source,'-o',exe])
        self.command('generated-list-run',[exe])
        self.assert_package_unchanged()

    def test_b_incomplete_abi_modes_and_invalid_overrides(self):
        cases=[]
        for i,path in enumerate(['src/runtime/gc.h','obj/runtime/gc.o','bin/nano_aot_runtime.o',
                                 *['bin/'+name for name in ('nanoc','nanoc_c','nanoc_stage1','nano_virt','nano_vm','nano_cop','nano_vmd','nanoisa','nvm2c')]]):
            cases.append(('missing-'+str(i),self.fault_generation('missing-'+str(i),lambda root,p=path:(root/p).unlink())))
        def abi(root,duplicate=False):
            p=root/'src/runtime/dyn_array.h';data=p.read_bytes()
            data=data+b'\n#define NANO_DYN_ARRAY_ABI_VERSION 2u\n' if duplicate else data.replace(b'#define NANO_DYN_ARRAY_ABI_VERSION 2u',b'#define NANO_DYN_ARRAY_ABI_VERSION 1u')
            replace(p,data)
        cases += [('abi1',self.fault_generation('abi1',abi)),('duplicate-abi',self.fault_generation('duplicate-abi',lambda root:abi(root,True)))]
        def noexecute(root):
            p=root/'bin/nanoc';replace(p,p.read_bytes(),0o644)
        cases.append(('non-executable',self.fault_generation('non-executable',noexecute)))
        def object_without_execute(root):
            p=root/'bin/nano_aot_runtime.o';replace(p,p.read_bytes(),0o644)
        object_generation=self.fault_generation('non-executable-aot-object',object_without_execute)
        self.command('non-executable-aot-object-probe',[self.probe,'root','0','installed'],
                     extra={'NANOLANG_SDK_ROOT':object_generation})
        cases += [('empty',''),('absent',self.work/'no-sdk')]
        type(self).validation_cases=cases
        program=self.outside/'invalid-control.nano';program.write_text('fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n')
        for name,generation in cases:
            self.command(name+'-probe',[self.probe,'root','1','installed'],extra={'NANOLANG_SDK_ROOT':generation})
            for compiler in ('nanoc_c','nanoc_stage1','nanoc'):
                binary=self.generation/'bin'/compiler
                output=self.outside/(name+'-'+compiler);output.write_bytes(b'original output')
                log=self.work/(name+'-'+compiler+'-cc.jsonl')
                _,_,state=self.command(name+'-'+compiler,[binary,program,'-o',output],extra={
                    'NANOLANG_SDK_ROOT':generation,'CC':self.observer,'NANO_CC':self.observer,
                    'SDK_CC_LOG':log,'SDK_REAL_CC':json.dumps(self.cc)},expected=(1,),timeout=900)
                self.assertEqual(state['returncode'],1);self.assertEqual(output.read_bytes(),b'original output');self.assertFalse(log.exists())
        self.assert_package_unchanged()

    def selected(self, source):
        seen=set();names=[];inputs={}
        def visit(path):
            path=Path(path).resolve()
            if path in seen:return
            seen.add(path);text=path.read_text();inputs[str(path)]=digest(path)
            for name in re.findall(r'^(?:unsafe\s+)?(?:import|from|module)\s+"([^"\n]+)"',text,re.M):
                possibilities=([path.parent/name] if name.startswith(('./','../')) else
                    [path.parent/name,self.outside/name,self.outside/'modules'/name,
                     self.generation/'stdlib'/name,self.generation/'modules'/name,self.generation/name])
                target=next((p for p in possibilities if p.is_file()),None)
                self.assertIsNotNone(target,(path,name));visit(target)
            names.extend(re.findall(r'^shadow\s+([A-Za-z_][A-Za-z_0-9]*)\s*\{',text,re.M))
        visit(source)
        return names,inputs

    def compile_installed(self, label, compiler, source, extra=None, run_expected=(0,), run_diagnostic=None):
        output=self.outside/(label+'-program');log=self.work/(label+'-cc.jsonl')
        expected,inputs=self.selected(source)
        args=[self.generation/'bin'/compiler,source,'-o',output,'--keep-c']
        if compiler=='nanoc_c':args+=['--verbose','--llm-shadow-json',self.work/(label+'-shadows.json')]
        env=dict(CC=self.observer,NANO_CC=self.observer,SDK_CC_LOG=log,SDK_REAL_CC=json.dumps(self.cc))
        env.update(extra or {})
        out,err,_=self.command(label+'-build',args,extra=env,timeout=1800)
        if compiler=='nanoc_c':
            report=json.loads((self.work/(label+'-shadows.json')).read_text())
            self.assertTrue(report['completed'] and report['success']);self.assertEqual(report['failures'],[])
            self.assertEqual(report['test_count'],len(expected))
            raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
        else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
        normalized=[re.sub(r'^__nano_module_+[0-9]+_','',value.decode()) for value in raw]
        self.assertCountEqual(normalized,expected)
        commands=[json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(commands)
        for argv in commands:
            self.assertFalse(any(str(ROOT) in arg or str(ROOT)+'-sdk-hidden' in arg for arg in argv),argv)
        self.assertTrue(any(str(self.generation) in arg for argv in commands for arg in argv))
        (self.work/(label+'-proof.json')).write_text(json.dumps(dict(inputs=inputs,expected=expected,actual=normalized,commands=commands,output=digest(output)),indent=2)+'\n')
        run_args=[output]
        if run_diagnostic is not None:
            launch=self.work/(label+'-no-core.py')
            launch.write_text('import os,resource,sys\nresource.setrlimit(resource.RLIMIT_CORE,(0,0))\nos.execv(sys.argv[1],sys.argv[1:])\n')
            run_args=[sys.executable,launch,output]
        out,err,_=self.command(label+'-run',run_args,expected=run_expected)
        if run_diagnostic is not None:
            self.assertIn(run_diagnostic,err)
            self.assertNotIn(b'AddressSanitizer:',err)
            self.assertNotIn(b'runtime error:',err)
        return output

    def test_c_installed_paired_programs_readonly_and_concurrency(self):
        module=self.outside/'helper.nano'
        module.write_text('pub fn answer() -> int { return 37 }\nshadow answer { assert (== (answer) 37) }\n')
        source=self.outside/'ordinary.nano'
        source.write_text('module "./helper.nano" as Local\nmodule "modules/std/json/json.nano" as Json\n'
            'struct Wide { '+', '.join('f'+str(i)+': int' for i in range(63))+' }\n'
            'fn main() -> int { let value: Wide = Wide { '+', '.join('f'+str(i)+': '+str(i) for i in range(63))+' }\n'
            'let values: array<Wide> = [value]\nlet result: Wide = (at values 0)\nassert (== result.f62 62)\n'
            'assert (== (Local.answer) 37)\nlet object: Json.Json = (Json.parse "{}")\n'
            'assert (Json.is_object object)\n(Json.free object)\nreturn 0 }\nshadow main { assert (== (main) 0) }\n')
        modes={p:stat.S_IMODE(p.stat().st_mode) for p in [self.generation,*self.generation.rglob('*')]}
        for p,mode in modes.items():p.chmod(mode & ~0o222)
        try:
            with self.hidden_source():
                for compiler in ('nanoc_c','nanoc_stage1','nanoc'):
                    self.compile_installed('readonly-'+compiler,compiler,source)
                # Both invocations select the same immutable SDK but separate work.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=[pool.submit(self.compile_installed,'concurrent-'+name,name,source)
                             for name in ('nanoc_c','nanoc')]
                    for result in results:result.result()
                link=self.outside/'via PATH';link.mkdir();(link/'nanoc').symlink_to(self.prefix/'bin/nanoc')
                out,_,_=self.command('path-and-symlink-abi',['nanoc','--native-array-abi'],extra={'PATH':str(link)+os.pathsep+os.environ['PATH']})
                self.assertEqual(out,b'2\n')
        finally:
            for p,mode in modes.items():p.chmod(mode)
        self.assert_package_unchanged()

    def test_d_installed_all_provider_owners_and_dynamic_wrappers(self):
        from tests import file_cseed_provider_owners as owners
        dynamic=self.work/'isolated-owner-host.py';shutil.copyfile(ROOT/'tests/file_cseed_provider_owners.py',dynamic)
        forth=self.outside/'forth.nano';shutil.copyfile(ROOT/'examples/language/nl_forth_interpreter.nano',forth)
        bytecode=self.outside/'forth.nvm'
        test=self
        class Adapter:
            work=test.work
            cc=test.cc
            flags=[*test.flags,'-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(test.generation/'src')]
            def __getattr__(self,name):return getattr(test,name)
            def command(self,name,args,timeout=180,extra=None):
                out,err,_=test.command('installed-owner-'+name,args,extra=extra,timeout=timeout)
                return out,err
        with self.hidden_source():
            self.command('installed-forth-bytecode',[self.prefix/'bin/nano_virt',forth,'--emit-nvm','-o',bytecode],timeout=1800)
            owners.run(Adapter(),self.generation,self.selected,installed={'forth_binary':bytecode,'dynamic_runner':dynamic})
        summary=json.loads((self.work/'public-provider-owners/summary.json').read_text())
        self.assertEqual(len(summary),36)
        for row in summary:
            proof=json.loads((self.work/'public-provider-owners'/(row['compiler']+'-'+row['case'])/'actual-artifacts.json').read_text())
            self.assertFalse(any(str(ROOT) in arg for arg in proof['final_command']))
        self.assert_package_unchanged()

    def test_d_overbudget_final_command_preserves_output(self):
        source=self.outside/'command-budget.nano'
        source.write_text('fn main() -> int { return 37 }\nshadow main { assert (== (main) 37) }\n')
        output=self.outside/'command-budget-output';sentinel=b'prior final output\n'
        output.write_bytes(sentinel)
        log=self.work/'command-budget-cc.jsonl';shadows=self.work/'command-budget-shadows.json'
        extra=dict(CC=self.observer,NANO_CC=self.observer,SDK_CC_LOG=log,
                   SDK_REAL_CC=json.dumps(self.cc),NANO_CFLAGS='-DSDK_COMMAND_BUDGET='+('0'*65536))
        with self.hidden_source():
            out,err,_=self.command('command-budget-refusal',
                [self.generation/'bin/nanoc_c',source,'-o',output,'--verbose','--keep-c',
                 '--llm-shadow-json',shadows],extra=extra,expected=(1,),timeout=1800)
        self.assertEqual(output.read_bytes(),sentinel)
        self.assertIn(b'C compile command too long',err)
        match=re.search(rb'arguments \((\d+) command bytes, limit (\d+)\)',err)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match[1]),65536);self.assertEqual(int(match[2]),65536)
        report=json.loads(shadows.read_text())
        self.assertTrue(report['completed'] and report['success'])
        self.assertEqual(report['failures'],[]);self.assertEqual(report['test_count'],1)
        self.assertEqual(re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M),[b'main'])
        commands=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        self.assertFalse(any(str(output) in argv for argv in commands),commands)
        self.assert_package_unchanged()

    def test_e_installed_abi_before_foreign_entry(self):
        directory=self.outside/'array abi';directory.mkdir()
        foreign=directory/'foreign.c'
        foreign.write_text('#include "runtime/dyn_array.h"\n#include <stdio.h>\n'
            '#if !defined(ABI_VERSION) || ABI_VERSION == 1\n'
            'typedef struct { int64_t length,capacity; ElementType elem_type; uint8_t elem_size; void *data; } FixtureArray;\n'
            '#else\ntypedef DynArray FixtureArray;\n#endif\n'
            'static FixtureArray value={.elem_type=ELEM_INT,.elem_size=8};\n'
            'FixtureArray *probe(void) { puts("foreign entered"); fflush(stdout); return &value; }\n'
            '#ifdef ABI_VERSION\nconst uint32_t probe__nano_array_abi=ABI_VERSION;\n#endif\n')
        source=directory/'main.nano'
        source.write_text('extern fn probe() -> array<int>\n'
            'fn main() -> int { unsafe { let values: array<int> = (probe) return (array_length values) } }\n')
        with self.hidden_source():
            for compiler in ('nanoc_c','nanoc_stage1','nanoc'):
                for version in (None,1,2,99):
                    for static in (False,True):
                        name=f'installed-abi-{compiler}-{version}-{int(static)}'
                        library=directory/('libfixture.dylib' if sys.platform=='darwin' else 'libfixture.so')
                        archive=directory/'libfixture.a'
                        library.unlink(missing_ok=True);archive.unlink(missing_ok=True)
                        object_file=directory/'foreign.o'
                        flags=['-c'] if static else (['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC'])
                        self.command(name+'-provider',[*self.cc,*self.flags,'-D_GNU_SOURCE','-I',self.generation/'src',*flags,
                            *([] if version is None else [f'-DABI_VERSION={version}']),foreign,'-o',object_file if static else library])
                        if static:self.command(name+'-archive',['ar','rcs',archive,object_file])
                        output=directory/name
                        args=[self.generation/'bin'/compiler,source,'-o',output,'--keep-c']
                        env={}
                        if compiler=='nanoc_c':args+=['-L',directory,'-lfixture']
                        else:env['NANO_LDFLAGS']='-L'+shlex.quote(str(directory))+' -lfixture'
                        self.command(name+'-build',args,extra=env,timeout=900)
                        # Missing/ABI1/unknown providers actually expose the old
                        # narrow layout or an incompatible marker. No entry is
                        # permitted; SIGABRT is the existing exact guard terminal.
                        out,err,state=self.command(name+'-run',[output],expected=(0,) if version==2 else (-6,),
                            extra={'LD_LIBRARY_PATH':directory,'DYLD_LIBRARY_PATH':directory})
                        if version==2:self.assertIn(b'foreign entered',out)
                        else:
                            self.assertNotIn(b'foreign entered',out);self.assertIn(b'native array ABI',err)
                        retained=self.work/name;retained.mkdir()
                        for p in (foreign,source,archive if static else library):shutil.copyfile(p,retained/p.name)
                        (self.work/(name+'-products.json')).write_text(json.dumps({str(p):digest(p)
                            for p in (foreign,source,output,archive if static else library)},indent=2)+'\n')
        self.assert_package_unchanged()

    def test_f_installed_standalone_and_daemon_wrappers(self):
        source=self.outside/'wrapper.nano'
        source.write_text('fn main() -> int { return 42 }\nshadow main { assert (== (main) 42) }\n')
        supervisor=self.outside/'daemon-supervisor.py'
        supervisor.write_text('import json,os,subprocess,sys,time\nfrom pathlib import Path\n'
            'daemon,program,report=sys.argv[1:];report=Path(report)\n'
            'out=(report/"daemon.stdout").open("wb");err=(report/"daemon.stderr").open("wb")\n'
            'p=None;state={}\ntry:\n'
            ' p=subprocess.Popen([daemon,"--foreground","--idle-timeout","30"],stdout=out,stderr=err)\n'
            ' deadline=time.monotonic()+10\n'
            ' while not Path(os.environ["NANOVMD_SOCKET"]).exists():\n'
            '  assert p.poll() is None,"daemon exited before socket"\n'
            '  assert time.monotonic()<deadline,"daemon startup deadline"\n'
            '  time.sleep(.02)\n'
            ' state["wrapper_returncode"]=subprocess.run([program],timeout=20).returncode\n'
            ' assert state["wrapper_returncode"]==42,state\n'
            'finally:\n'
            ' if p is not None:\n'
            '  if p.poll() is None:p.terminate()\n'
            '  try:state["daemon_returncode"]=p.wait(timeout=5)\n'
            '  except subprocess.TimeoutExpired:p.kill();state["daemon_returncode"]=p.wait(timeout=5);state["forced_kill"]=True\n'
            ' out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno());out.close();err.close()\n'
            ' (report/"daemon-status.json").write_text(json.dumps(state)+"\\n")\n'
            ' assert not state.get("forced_kill"),state\n')
        with self.hidden_source():
            for daemon in (False,True):
                name='daemon' if daemon else 'standalone'
                output=self.outside/('installed-'+name)
                log=self.work/(name+'-wrapper-cc.jsonl')
                self.command(name+'-wrapper-build',[self.prefix/'bin/nano_virt',source,'-o',output,
                    *(['--daemon-wrapper'] if daemon else [])],extra={
                        'CC':self.observer,'NANO_CC':self.observer,'SDK_CC_LOG':log,'SDK_REAL_CC':json.dumps(self.cc)},timeout=900)
                commands=[json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(commands),2,commands)
                compile_args,link_args=commands
                self.assertIn('-c',compile_args);self.assertNotIn('-c',link_args)
                self.assertEqual(compile_args.count('-o'),1);self.assertEqual(link_args.count('-o'),1)
                wrapper_object=Path(compile_args[compile_args.index('-o')+1])
                stage=wrapper_object.parent
                self.assertEqual(wrapper_object.name,'wrapper.o')
                self.assertEqual(stage.parent,self.outside)
                self.assertRegex(stage.name,r'^\.nano-wrapper-[A-Za-z0-9]{6}$')
                self.assertEqual([arg for arg in compile_args if arg.endswith('.c')],[str(stage/'source.c')])
                self.assertEqual(link_args[link_args.index('-o')+1],str(stage/'executable'))
                objects=[arg for arg in link_args if arg.endswith('.o')]
                self.assertEqual(objects.count(str(wrapper_object)),1)
                providers=[item for item in objects if item!=str(wrapper_object)]
                self.assertTrue(providers)
                for item in providers:self.assertTrue(Path(item).is_relative_to(self.generation/'obj'),item)
                self.assertFalse(stage.exists())
                if daemon:
                    socket_dir=Path(tempfile.mkdtemp(prefix='sdk-vmd-',dir='/tmp'))
                    sock=socket_dir/'vm.sock'
                    self.command('daemon-wrapper-run',[sys.executable,supervisor,self.prefix/'bin/nano_vmd',output,self.work],
                        extra={'NANOVMD_SOCKET':sock},timeout=60)
                    self.assertFalse(sock.exists())
                    self.assertFalse((socket_dir/'vm.sock.pid').exists())
                    socket_dir.rmdir()
                else:self.command('standalone-wrapper-run',[output],expected=(42,))
        self.assert_package_unchanged()

    def test_g_user_metadata_required_provider_failure(self):
        project=self.outside/'metadata project with spaces';project.mkdir()
        directory=project/'user';directory.mkdir()
        include=directory/'preferred include';include.mkdir()
        fallback=project/'fallback';fallback.mkdir()
        for header_dir,value in ((include,37),(fallback,91)):
            (header_dir/'answer.h').write_text('#include <stdint.h>\n#define SDK_HEADER_ANSWER '+str(value)+'\nint64_t sdk_user_answer(void);\n')
        provider=directory/'answer.c'
        provider.write_text('#include "answer.h"\n#if SDK_TRUSTED_FLAG != 1\n#error I require the unchanged trusted flag\n#endif\nint64_t sdk_user_answer(void) { return SDK_HEADER_ANSWER; }\n')
        (directory/'module.json').write_text(json.dumps(dict(name='sdk_user_metadata',
            headers=['answer.h'],c_sources=['answer.c'],include_dirs=[str(include)],
            cflags=['-Ifallback','-DSDK_TRUSTED_FLAG=1']),indent=2)+'\n')
        module=directory/'user.nano'
        module.write_text('pub extern fn sdk_user_answer() -> int\n')
        source=self.outside/'metadata.nano'
        source.write_text('module "./metadata project with spaces/user/user.nano" as User\n'
            'module "modules/std/json/json.nano" as Json\n'
            'fn main() -> int { unsafe { assert (== (User.sdk_user_answer) 37) }\n'
            'let object: Json.Json = (Json.parse "{}")\nassert (Json.is_object object)\n(Json.free object)\nreturn 0 }\n'
            'shadow main { assert (== (main) 0) }\n')
        with self.hidden_source():
            for compiler in ('nanoc_c','nanoc_stage1','nanoc'):
                self.compile_installed('mixed-origin-'+compiler,compiler,source)
                proof=json.loads((self.work/('mixed-origin-'+compiler+'-proof.json')).read_text())
                provider_rows=[row for row in proof['commands'] if str(provider) in row]
                self.assertTrue(provider_rows)
                matched=[]
                for row in provider_rows:
                    includes=[];i=0
                    while i<len(row):
                        arg=row[i]
                        if arg=='-I':
                            self.assertLess(i+1,len(row));i+=1;includes.append(row[i])
                        elif arg.startswith('-I'):includes.append(arg[2:])
                        i+=1
                    if str(include) in includes and str(fallback) in includes:
                        self.assertLess(includes.index(str(include)),includes.index(str(fallback)))
                        self.assertIn(str(self.generation/'src'),includes)
                        self.assertLess(includes.index(str(include)),includes.index(str(self.generation/'src')))
                        self.assertIn('-DSDK_TRUSTED_FLAG=1',row)
                        self.assertNotIn('with',row);self.assertNotIn('spaces/fallback',row)
                        matched.append(dict(argv=row,includes=includes))
                self.assertTrue(matched,(compiler,provider_rows))
                (self.work/('metadata-include-argv-'+compiler+'.json')).write_text(json.dumps(matched,indent=2)+'\n')
                self.assertTrue(any(str(self.generation/'src/cJSON.c') in row for row in proof['commands']))
        # I model an actual selected compiler failure only for the required user
        # provider; other commands retain the real selected compiler and argv.
        refuse=self.work/'refuse-provider.py'
        refuse.write_text('#!'+sys.executable+'\nimport json,os,subprocess,sys\n'
            'from pathlib import Path\n'
            'if os.environ["SDK_REFUSED_SOURCE"] in sys.argv[1:]:\n'
            ' Path(os.environ["SDK_REFUSAL_HIT"]).write_text(json.dumps(sys.argv[1:])+"\\n")\n'
            ' sys.exit(73)\n'
            'sys.exit(subprocess.call(json.loads(os.environ["SDK_REAL_CC"])+sys.argv[1:]))\n')
        refuse.chmod(0o755)
        with self.hidden_source():
            for compiler in ('nanoc_c','nanoc_stage1','nanoc'):
                output=self.outside/('provider-failure-'+compiler);output.write_bytes(b'original output')
                hit=self.work/('provider-refusal-'+compiler+'.json')
                cache=self.private/('failed-cache-'+compiler)
                self.command('required-provider-refusal-'+compiler,[self.generation/'bin'/compiler,source,'-o',output],
                    extra={'CC':refuse,'NANO_CC':refuse,'SDK_REFUSED_SOURCE':provider,'SDK_REFUSAL_HIT':hit,
                           'SDK_REAL_CC':json.dumps(self.cc),'NANO_BUILD_CACHE':cache},expected=(1,),timeout=900)
                self.assertTrue(hit.is_file());self.assertEqual(output.read_bytes(),b'original output')
        self.assert_package_unchanged()

    def test_h_selected_discovery_instrumentation_matrix(self):
        # Installed compilers and wrapper providers remain the same ordinary
        # generation. I freshly build only the root/private-work owning TU and
        # probe for each selected compiler/instrumentation configuration.
        configs=json.loads(os.environ.get('NANO_SDK_PROBE_CONFIGS','[]'))
        self.assertTrue(configs,'I require an explicit host probe matrix')
        for row in configs:
            name=row['name'];probe=self.work/('probe-'+name)
            flags=['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',ROOT/'src']
            if row['sanitizers']:flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
            self.command(name+'-probe-build',[*row['cc'],*self.flags,*flags,ROOT/'tests/native_sdk_probe.c',
                ROOT/'src/runtime/module_build_dir.c',*self.links,'-o',probe])
            for mode,expected in (('root',0),('short',3),('objects',0),('lists',0)):
                out,_,_=self.command(name+'-'+mode,[probe,mode,str(expected),'installed'],
                    extra={'NANOLANG_SDK_ROOT':self.generation})
                if mode=='objects':
                    for key in ('FIRST','SECOND','CACHE'):
                        self.assertFalse(Path(spans(out)[key]).exists())
                if mode=='lists':
                    directory=Path(spans(out)['FIRST'])
                    self.assertTrue((directory/'list_SdkPair.c').is_file())
                    self.assertTrue((directory/'list_SdkPair.h').is_file())
            for label,generation in self.validation_cases:
                self.command(name+'-'+label,[probe,'root','1','installed'],extra={'NANOLANG_SDK_ROOT':generation})
        self.assert_package_unchanged()

    def test_i_actual_generator_failure_and_partial_install(self):
        def failing_script(root):
            script=root/'scripts/generate_list.sh'
            replace(script,b'#!/bin/sh\nprintf "I modeled a required generator failure.\\n" >&2\nexit 73\n')
        generation=self.fault_generation('generator-failure',failing_script)
        # The complete valid inventory is self-consistent; this is an explicit
        # modeled tool failure, not an incomplete-package/root-refusal control.
        self.command('generator-failure-root',[self.probe,'root','0','installed'],extra={'NANOLANG_SDK_ROOT':generation})
        source=self.outside/'required-list.nano'
        source.write_text('struct SdkPair { value: int }\n'
            'fn main() -> int { let values: List<SdkPair> = (list_SdkPair_new)\n'
            'let value: SdkPair = SdkPair { value: 37 }\n(list_SdkPair_push values value)\n'
            'let result: SdkPair = (list_SdkPair_get values 0)\nreturn (- result.value 37) }\n')
        output=self.outside/'required-list';output.write_bytes(b'original output')
        with self.hidden_source():
            out,err,_=self.command('required-generator-refusal',[generation/'bin/nanoc_c',source,'-o',output],expected=(1,),timeout=900)
        self.assertIn(b'I modeled a required generator failure.',out+err)
        self.assertEqual(output.read_bytes(),b'original output')
        partial=self.work/'partial prefix';partial.mkdir();(partial/'sentinel').write_bytes(b'user input')
        script=self.work/'partial-install.py'
        script.write_text('import errno,json,os,stat,sys\nfrom pathlib import Path\n'
            'sys.path.insert(0,sys.argv[1]+"/scripts")\nimport native_sdk as sdk\n'
            'original=sdk.os.fsync;hits=[]\n'
            'def fail(fd):\n'
            ' if stat.S_ISREG(os.fstat(fd).st_mode):\n'
            '  hits.append(fd);raise OSError(errno.EIO,"modeled copied-file sync failure")\n'
            ' return original(fd)\n'
            'sdk.os.fsync=fail\n'
            'try:\n'
            ' try:sdk.install(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[1])/"scripts/native_sdk_inputs.json")\n'
            ' except RuntimeError as e:assert isinstance(e.__cause__,OSError) and e.__cause__.errno==errno.EIO and len(hits)==1 and "generation_committed=false" in str(e),(e,hits)\n'
            ' else:raise AssertionError("partial install unexpectedly succeeded")\n'
            'finally:sdk.os.fsync=original\n'
            'Path(sys.argv[3]).write_text(json.dumps({"modeled_first_regular_file_fsync_failure":len(hits)})+"\\n")\n')
        self.command('partial-install',[sys.executable,script,ROOT,partial,self.work/'partial-install-result.json'],timeout=900)
        self.assertEqual((partial/'sentinel').read_bytes(),b'user input')
        parent=partial/'lib/nanolang/sdk'
        self.assertFalse(any(parent.iterdir()) if parent.exists() else False)
        self.assertFalse(any((partial/'bin').iterdir()) if (partial/'bin').exists() else False)
        self.assert_package_unchanged()

    def test_j_installed_dependency_header_origins(self):
        from tests import file_module_dependency_headers as headers
        test=self
        class Adapter:
            work=test.work
            cc=test.cc
            def __getattr__(self,name):return getattr(test,name)
            def command(self,name,args,timeout=180,extra=None):
                out,err,_=test.command('installed-'+name,args,timeout=timeout,extra=extra)
                return out,err
        with self.hidden_source():
            headers.run(Adapter(),self.generation,installed=True)
        self.assert_package_unchanged()

    def test_k_installed_opaque_and_exact_carriers(self):
        from tests import native_sdk_opaque_cases
        with self.hidden_source():
            native_sdk_opaque_cases.run(self)
        self.assert_package_unchanged()

    def test_y_installer_special_inputs_and_owned_boundaries(self):
        stage=self.work/'special-sdk';stage.mkdir();os.mkfifo(stage/'sdk.inputs')
        self.command('fifo-manifest',[sys.executable,ROOT/'scripts/native_sdk.py','verify','--root',stage],expected=(1,),timeout=15)
        unknown=self.generation/'unknown-empty';unknown.mkdir()
        try:
            self.command('unknown-generation-entry',[sys.executable,ROOT/'scripts/native_sdk.py','verify','--root',self.generation],expected=(1,))
        finally:unknown.rmdir()
        external=self.work/'external-bin';external.mkdir()
        bin_dir=self.prefix/'bin';saved=self.prefix/'owned-bin';bin_dir.rename(saved);bin_dir.symlink_to(external,target_is_directory=True)
        outside=external/'nanoc';outside.symlink_to('../lib/nanolang/sdk/'+self.generation.name+'/bin/nanoc')
        try:
            self.command('uninstall-symlink-bin',[sys.executable,ROOT/'scripts/native_sdk.py','uninstall','--prefix',self.prefix],expected=(1,))
            self.assertTrue(outside.is_symlink());self.assert_package_unchanged()
        finally:bin_dir.unlink();saved.rename(bin_dir)

    def test_z_actual_owned_uninstall(self):
        (self.prefix/'unrelated').write_bytes(b'user sentinel')
        (self.generation/'unowned-empty').mkdir();(self.generation/'unowned-file').write_bytes(b'keep')
        self.command('actual-uninstall',['make','PREFIX='+str(self.prefix),'uninstall'],cwd=ROOT)
        self.assertEqual((self.prefix/'unrelated').read_bytes(),b'user sentinel')
        self.assertTrue((self.generation/'unowned-empty').is_dir());self.assertEqual((self.generation/'unowned-file').read_bytes(),b'keep')
        for row in self.rows:self.assertFalse(os.path.lexists(self.generation/row['path']))
        for name in sdk.PUBLIC:self.assertFalse(os.path.lexists(self.prefix/'bin'/name))
        self.assertFalse((self.prefix/'lib/libnano_file_runtime.a').exists())
        self.assertFalse(any((self.prefix/'include/nanolang/file').rglob('*.h')))
        self.assertFalse((self.generation/'sdk.inputs').exists())
