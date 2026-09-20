"""I qualify retained declaration syntax, not File source execution.

I require a fresh bootstrap/ABI closure supplied by the outer gate. Sanitizers
cover the included parser fixture plus freshly compiled env/lexer/UTF8 providers;
all named common/runtime providers are retained ordinary objects.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import unittest
from tests import test_file_source_plan as retained_runner
from tests import test_token_value_bytes as retained_schema
ROOT=Path(__file__).resolve().parents[1]

class FileServiceParser(unittest.TestCase):
    command=classmethod(retained_runner.FileSourcePlan.command.__func__)
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-file-service-parser-',dir=os.environ.get('NANO_SERVICE_PARSER_REPORT_DIR')))
        print('I retain parser artifacts at',cls.work,flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_SERVICE_PARSER_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_SERVICE_PARSER_CFLAGS',''))+['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(ROOT/'src')]
        cls.links=shlex.split(os.environ.get('NANO_SERVICE_PARSER_LDFLAGS',''))
        if os.environ.get('NANO_SERVICE_PARSER_SANITIZERS','0')=='1':
            cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.common=[Path(p).resolve() for p in shlex.split(os.environ['NANO_SERVICE_PARSER_OBJECTS'])]
        if not cls.common or any(p.name in ('parser.o','env.o','lexer.o','utf8.o','main.o') for p in cls.common):
            raise AssertionError('I require exact ordinary common/runtime objects excluding selected rebuilt TUs')
        cls.before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.common}
        (cls.work/'ordinary-providers-before.json').write_text(json.dumps(cls.before,indent=2)+'\n')
        cls.selected=[]
        for provider in ('lexer','utf8','env'):
            observer=['-include',str(ROOT/'tests/file_service_parser_free_hooks.h'),'-DNANO_FILE_SERVICE_PARSER_OBSERVE_FREE=1'] if provider=='env' else []
            obj=cls.work/(provider+'.o');cls.command(provider+'-build',[*cls.cc,*cls.flags,*observer,'-c',ROOT/'src'/(provider+'.c'),'-o',obj]);cls.selected.append(obj)
        # I build and invoke the actual installed-tool source recipe, not a copied golden renderer.
        cls.publisher=cls.work/'publisher-bin/nsi-file-binding'
        cls.command('publisher-make',[shutil.which('make'),'-f','Makefile.gnu','-j2','CC='+shlex.join(cls.cc),
            'CFLAGS='+shlex.join(cls.flags),'LDFLAGS='+shlex.join(cls.links),
            'OBJ_DIR='+str(cls.work/'publisher-obj'),'BIN_DIR='+str(cls.publisher.parent),'nsi-file-binding'],timeout=300)
        cls.command('publish',[cls.publisher,ROOT/'tests/fixtures/nsi_file_plan.json','--file-binding-dir',cls.work/'published'])
        cls.binding=cls.work/'published/binding.nano'
        if cls.binding.read_bytes()!=(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_bytes():
            raise AssertionError('actual publisher source differs from retained complete golden')
        fresh=cls.selected+[cls.publisher]+sorted((cls.work/'publisher-obj').rglob('*.o'))
        (cls.work/'fresh-providers.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in fresh},indent=2)+'\n')
        (cls.work/'scope.json').write_text(json.dumps({'parser':'included and allocation-hooked','fresh_selected':['lexer','utf8','env'], 'env_observer':'actual owning-TU free only; all real frees execute; env allocations are not parser failure-prefix hooks',
            'common_providers':'ordinary retained exact list','publisher':'fresh seven-provider actual Make closure',
            'service_execution':False,'bootstrap':'required fresh outer prerequisite'},indent=2)+'\n')
    @classmethod
    def tearDownClass(cls):
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.common}
        (cls.work/'ordinary-providers-after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=cls.before:raise AssertionError('ordinary providers changed during parser gate')
    def test_c_ownership_and_refusal(self):
        exe=self.work/'parser-c'
        self.command('parser-c-build',[*self.cc,*self.flags,ROOT/'tests/test_file_service_parser.c',*self.selected,*self.common,*self.links,'-o',exe])
        out,_=self.command('parser-c-run',[exe,self.binding])
        lines=out.decode().splitlines()
        self.assertEqual(lines[-1],'publisher:1:5:retained:consumers-refused')
        rows=[line for line in lines if line.startswith('allocation:')]
        self.assertEqual(len(lines),len(rows)+2)
        self.assertEqual(lines[-2],'ordinary-bindings:zero:named:discard:payload-refused:unknown-refused:backends-checked')
        self.assertGreater(len(rows),100)
        self.assertEqual(len(rows)%2,0)
        total=len(rows)//2
        self.assertGreater(total,100)
        self.assertEqual(rows,[f'allocation:{mode}:{i}:refused:recovered' for mode in ('transient','persistent') for i in range(1,total+1)])
    def test_paired_compilers_schema_and_consumers(self):
        # I retain the existing complete token/helper and independent actual
        # generator corpus: all three producers, exact shadow multisets, all4 files.
        retained_schema.TokenValueBytes.test_paired_fresh_compilers(self)
        source=self.work/'file_service_parser.nano'
        template=(ROOT/'tests/file_service_parser.nano.in').read_text()
        self.assertEqual(template.count('@PUBLISHER_SOURCE@'),1)
        source.write_text(template.replace('@PUBLISHER_SOURCE@',json.dumps(self.binding.read_text(),ensure_ascii=False)))
        expected=[];seen=set();inputs={}
        def visit(path):
            path=path.resolve()
            if path in seen:return
            seen.add(path);text=path.read_text();inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for name in re.findall(r'^(?:unsafe\s+)?(?:import|from|module)\s+"([^"\n]+)"',text,re.M):
                target=next((p for p in (ROOT/name,path.parent/name,ROOT/'modules'/name) if p.is_file()),None)
                if target is None:raise AssertionError(('missing selected import',path,name))
                visit(target)
            expected.extend(re.findall(r'^shadow\s+([A-Za-z_][A-Za-z_0-9]*)\s*\{',text,re.M))
        visit(source)
        (self.work/'parser-expected-selection.json').write_text(json.dumps({'names':expected,'inputs':inputs},indent=2)+'\n')
        for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/('parser-'+name);report=self.work/(name+'-shadows.json');tmp=self.work/(name+'-tmp');tmp.mkdir()
            args=[ROOT/'bin'/name,source,'-o',exe,'--keep-c']
            if name=='nanoc_c':args+=['--llm-shadow-json',report,'--verbose']
            out,err=self.command(name+'-build',args,timeout=900,extra={'NANO_SHADOW_TRACE':'1','TMPDIR':str(tmp)})
            if name=='nanoc_c':
                data=json.loads(report.read_text());self.assertTrue(data['completed'] and data['success']);self.assertEqual(data['failures'],[]);self.assertEqual(data['test_count'],len(expected))
                raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
            else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            normalized=[re.sub(r'^__nano_module_+[0-9]+_','',p.decode()) for p in raw]
            (self.work/(name+'-selection.json')).write_text(json.dumps({'raw':[p.decode() for p in raw],'normalized':normalized,'expected':expected},indent=2)+'\n')
            self.assertCountEqual(normalized,expected)
            output,_=self.command(name+'-run',[exe]);self.assertEqual(output,b'publisher:1:5:retained\n')
        # I exercise both actual frontends on ordinary source, without any
        # service declaration whose guard could mask arity checking.
        ordinary_prefix='union Choice { None {}, Some { value: int } } fn main() -> int { match Choice.None {} { '
        ordinary_cases=[('zero','None() => { return 7 } Some(v) => { return v.value }',True),
            ('named-discard','None(n) => { return 7 } Some(_) => { return 8 }',True),
            ('payload-omitted','None() => { return 7 } Some() => { return 8 }',False),
            ('unknown','Missing() => { return 7 } Some(v) => { return v.value }',False),
            ('malformed','None() => { return 7 } Some(,) => { return 8 }',False)]
        for label,arms,accepted in ordinary_cases:
            ordinary=self.work/('ordinary-'+label+'.nano');ordinary.write_text(ordinary_prefix+arms+' } }\n')
            for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                executable=self.work/('ordinary-'+label+'-'+name)
                argv=[ROOT/'bin'/name,ordinary,'-o',executable,'--keep-c']
                if accepted:
                    self.command(label+'-'+name+'-build',argv,timeout=900)
                    status=self.work/(label+'-'+name+'-run.json')
                    script="import subprocess,sys,json,pathlib; p=subprocess.run(sys.argv[2:],timeout=60); pathlib.Path(sys.argv[1]).write_text(json.dumps({'returncode':p.returncode})); sys.exit(0 if p.returncode==7 else 1)"
                    self.command(label+'-'+name+'-run',[sys.executable,'-c',script,status,executable],timeout=80)
                    self.assertEqual(json.loads(status.read_text())['returncode'],7)
                else:
                    status=self.work/(label+'-'+name+'-refusal.json')
                    script="import subprocess,sys,json,pathlib; p=subprocess.run(sys.argv[2:],timeout=120); pathlib.Path(sys.argv[1]).write_text(json.dumps({'returncode':p.returncode})); sys.exit(0 if p.returncode>0 else 1)"
                    out,err=self.command(label+'-'+name+'-refusal',[sys.executable,'-c',script,status,*argv],timeout=140)
                    self.assertFalse(executable.exists());self.assertNotIn(b'resolved File service',out+b'\n'+err)
                    if label=='payload-omitted':self.assertIn(b'zero-field',out+b'\n'+err)
        for name in ('nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt'):
            output=self.work/(name+'-forbidden-output');status=self.work/(name+'-refusal.json')
            script="import subprocess,sys,json,pathlib; p=subprocess.run(sys.argv[2:],timeout=120); pathlib.Path(sys.argv[1]).write_text(json.dumps({'returncode':p.returncode})); sys.exit(0 if p.returncode>0 else 1)"
            out,err=self.command(name+'-consumer-refusal',[sys.executable,'-c',script,status,ROOT/'bin'/name,self.binding,'-o',output],timeout=140)
            self.assertFalse(output.exists());self.assertIn(b'resolved File service',out+b'\n'+err)

def load_tests(loader,tests,pattern):
    def ids(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from ids(test)
            else:yield test.id()
    expected=[__name__+'.FileServiceParser.'+name for name in ('test_c_ownership_and_refusal','test_paired_compilers_schema_and_consumers')]
    if sorted(ids(tests))!=sorted(expected):raise AssertionError('I require exactly the two intended parser methods')
    return tests
if __name__=='__main__':unittest.main()
