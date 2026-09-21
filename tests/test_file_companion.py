"""I qualify descriptive source facts; no File source program gains execution."""
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
from tests.file_companion_corpus import corpus, report_source, provider_report_source
from tests import file_native_provider_corpus
from tests import file_cseed_provider_owners
ROOT=Path(__file__).resolve().parents[1]
SMALL=('src/file_companion_snapshot.c','src/file_companion_bridge.c','src/file_source_input.c',
       'src/nsi_file_binding.c','src/nsi_file_plan.c','src/nsi.c','src/cJSON.c','src/utf8.c',
       'src/nanoisa/file_source_catalog.c')

def digest(paths):
    return {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

def selected(source):
    seen=set();names=[];inputs={}
    def visit(path):
        path=path.resolve()
        if path in seen:return
        seen.add(path);text=path.read_text();inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in re.findall(r'^(?:unsafe\s+)?(?:import|from|module)\s+"([^"\n]+)"',text,re.M):
            target=next((p for p in (ROOT/name,path.parent/name,ROOT/'modules'/name) if p.is_file()),None)
            if target is None:raise AssertionError((path,name))
            visit(target)
        names.extend(re.findall(r'^shadow\s+([A-Za-z_][A-Za-z_0-9]*)\s*\{',text,re.M))
    visit(source)
    return names,inputs

def reports(data):
    # Compiler/parser diagnostics are retained separately; these tagged rows
    # contain complete counted bytes and original tuples from the actual APIs.
    return b'\n'.join(line for line in data.splitlines() if line.startswith(
        (b'STATUS ',b'COUNTS ',b'ORIGIN ',b'ROW ',b'PLAN ',b'SNAPSHOT ',b'COLLECTION_REFUSED')))+b'\n'

def decode_span(value):
    size,encoded=value.split(b':',1);raw=bytes.fromhex(encoded.decode());assert len(raw)==int(size);return raw

class FileCompanion(unittest.TestCase):
    command=classmethod(retained_runner.FileSourcePlan.command.__func__)
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-file-companion-',dir=os.environ.get('NANO_COMPANION_REPORT_DIR')))
        print('I retain companion artifacts at',cls.work,flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_COMPANION_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_COMPANION_CFLAGS',''))+['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(ROOT/'src')]
        cls.links=shlex.split(os.environ.get('NANO_COMPANION_LDFLAGS',''))
        if os.environ.get('NANO_COMPANION_SANITIZERS')=='1':cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.common=[Path(p).resolve() for p in shlex.split(os.environ['NANO_COMPANION_COMMON_OBJECTS'])]
        excluded={Path(p).stem+'.o' for p in SMALL}|{'main.o','file_source_resolution.o','file_source_plan.o'}
        if not cls.common or any(p.name in excluded for p in cls.common):raise AssertionError('I require ordinary closure excluding freshly selected providers')
        cls.before=digest(cls.common);(cls.work/'ordinary-before.json').write_text(json.dumps(cls.before,indent=2)+'\n')
        cls.objects={}
        for mode in ('linked','instrumented'):
            cls.objects[mode]=[]
            for source in SMALL:
                obj=cls.work/(mode+'-'+Path(source).stem+'.o')
                hook=['-include',str(ROOT/'tests/file_companion_hooks.h')] if mode=='instrumented' else []
                cls.command(mode+'-'+obj.stem,[*cls.cc,*cls.flags,*hook,'-c',ROOT/source,'-o',obj])
                cls.objects[mode].append(obj)
        cls.graph=[]
        for name in ('file_source_resolution','nanoisa/file_source_plan'):
            obj=cls.work/(Path(name).stem+'.o');cls.command('graph-'+obj.stem,[*cls.cc,*cls.flags,'-c',ROOT/'src'/(name+'.c'),'-o',obj]);cls.graph.append(obj)
        cls.cases=corpus(ROOT,cls.work)
        # Actual publisher recipe and full golden output, not rewritten service text.
        cls.publisher=cls.work/'publisher-bin/nsi-file-binding'
        cls.command('publisher-make',['make','-f','Makefile.gnu','-j2','CC='+shlex.join(cls.cc),
            'CFLAGS='+shlex.join(cls.flags),'LDFLAGS='+shlex.join(cls.links),
            'OBJ_DIR='+str(cls.work/'publisher-obj'),'BIN_DIR='+str(cls.publisher.parent),'nsi-file-binding'],timeout=600)
        cls.command('publisher-run',[cls.publisher,ROOT/'tests/fixtures/nsi_file_plan.json','--file-binding-dir',cls.work/'actual-published'])
        actual=(cls.work/'actual-published/binding.nano').read_bytes()
        if actual!=(ROOT/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_bytes():raise AssertionError('publisher golden mismatch')
        (cls.work/'publisher/main.nano').write_bytes(actual)
        (cls.work/'fresh-providers.json').write_text(json.dumps(digest([p for rows in cls.objects.values() for p in rows]+cls.graph),indent=2)+'\n')
    @classmethod
    def tearDownClass(cls):
        after=digest(cls.common);(cls.work/'ordinary-after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=cls.before:raise AssertionError('ordinary provider mutation')
    def test_input_snapshot_bridge(self):
        for mode in ('linked','instrumented'):
            directory=self.work/('low-'+mode);directory.mkdir();module=directory/'main.nano';module.write_text('')
            shutil.copyfile(ROOT/'tests/fixtures/nsi_file_plan.json',directory/'interface.nsi.json')
            exe=self.work/('low-'+mode+'.exe')
            options=['-DCOMPANION_INSTRUMENT'] if mode=='instrumented' else []
            self.command(mode+'-fixture-build',[*self.cc,*self.flags,*options,ROOT/'tests/test_file_companion.c',*self.objects[mode],*self.links,'-o',exe])
            out,_=self.command(mode+'-fixture-run',[exe,directory/'input.nano',module,directory/'interface.nsi.json'],timeout=900)
            self.assertIn(b'PASS companion checks=',out)
    def test_graph_allocation_prefixes(self):
        objects=[]
        for source in ('src/file_source_resolution.c','src/nanoisa/file_source_plan.c'):
            obj=self.work/('instrumented-'+Path(source).stem+'.o')
            self.command('graph-hooks-'+obj.stem,[*self.cc,*self.flags,'-include',ROOT/'tests/file_companion_hooks.h','-c',ROOT/source,'-o',obj]);objects.append(obj)
        directory=self.work/'graph-fault-input';directory.mkdir();(directory/'main.nano').write_text('')
        shutil.copyfile(ROOT/'tests/fixtures/nsi_file_plan.json',directory/'interface.nsi.json')
        exe=self.work/'graph-faults'
        self.command('graph-fault-build',[*self.cc,*self.flags,'-DCOMPANION_INSTRUMENT','-DCOMPANION_GRAPH',ROOT/'tests/test_file_companion.c',*objects,*self.objects['instrumented'],*self.common,*self.links,'-o',exe])
        out,_=self.command('graph-fault-run',[exe,directory/'input.nano',directory/'main.nano',directory/'interface.nsi.json',self.work/'publisher/main.nano'],timeout=1800)
        self.assertIn(b'GRAPH_ALLOCATION transient=0 index=1 ',out)
        self.assertIn(b'GRAPH_ALLOCATION transient=1 index=1 ',out)
        self.assertIn(b'PASS companion checks=',out)

    def test_complete_paired_graph_reports(self):
        c=self.work/'resolution-c'
        self.command('resolution-c-build',[*self.cc,*self.flags,ROOT/'tests/file_resolution_report.c',*self.graph,*self.objects['linked'],*self.common,*self.links,'-o',c])
        baseline={}
        for case in self.cases:
            out,_=self.command('c-'+case['name'],[c,case['entry']],extra=case['env'])
            result=reports(out);self.assertIn(f"STATUS {case['status']}\n".encode(),result);baseline[case['name']]=result
            if case['status']==1:
                names=[decode_span(line.split()[3]).decode() for line in result.splitlines() if line.startswith(b'ROW ')]
                for name in case['required']:self.assertIn(name,names)
                for name in case['absent']:self.assertNotIn(name,names)
                rows=[line.split() for line in result.splitlines() if line.startswith(b'ROW ')]
                by_id={int(row[6]):row for row in rows}
                for row in rows:
                    target=by_id[int(row[7])]
                    self.assertEqual(decode_span(row[4]),decode_span(target[1]))
                    self.assertEqual(decode_span(row[5]),decode_span(target[3]))
                if case['name']=='global-vs-local':
                    selected_ids={decode_span(row[3]).decode():int(row[6]) for row in rows}
                    self.assertEqual(selected_ids['value'],14);self.assertEqual(selected_ids['answer'],15)
                if case['name']=='ordinary-kinds':
                    facts={decode_span(row[3]).decode():(int(row[6]),int(row[8])) for row in rows}
                    for index,(name,kind) in enumerate((('Record',3),('Color',4),('Choice',5),('External',6),('answer',2),('top',7)),14):
                        self.assertEqual(facts[name],(index,kind))

        source=self.work/'resolution-probe.nano';source.write_text(report_source(ROOT))
        names,inputs=selected(source);(self.work/'expected-selection.json').write_text(json.dumps(dict(names=names,inputs=inputs),indent=2)+'\n')
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/(compiler+'-resolution');shadow=self.work/(compiler+'-shadows.json')
            args=[ROOT/'bin'/compiler,source,'-o',exe,'--keep-c']
            if compiler=='nanoc_c':args+=['--llm-shadow-json',shadow,'--verbose']
            out,err=self.command(compiler+'-probe-build',args,timeout=1800,extra={'NANO_SHADOW_TRACE':'1'})
            if compiler=='nanoc_c':
                report=json.loads(shadow.read_text());self.assertTrue(report['completed'] and report['success']);self.assertEqual(report['failures'],[]);self.assertEqual(report['test_count'],len(names))
                raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
            else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            normalized=[re.sub(r'^__nano_module_+[0-9]+_','',name.decode()) for name in raw]
            (self.work/(compiler+'-selection.json')).write_text(json.dumps(dict(raw=[n.decode() for n in raw],normalized=normalized,expected=names),indent=2)+'\n');self.assertCountEqual(normalized,names)
            for case in self.cases:
                out,_=self.command(compiler+'-'+case['name'],[exe,case['entry']],extra=case['env']);result=reports(out)
                if case['status'] in (2,3,5) and result==b'COLLECTION_REFUSED\n':
                    # Actual collector has only ok, not a recoverable status enum.
                    # I retain the C-specific refusal separately; neither publishes.
                    continue
                self.assertEqual(result,baseline[case['name']])


    def test_actual_opted_in_drivers_preserve_outputs(self):
        # The wrapper is an external observer. Preparatory service graphs must
        # return before any compiler invocation, including attempted failures.
        observer=self.work/'forbidden-compiler.py'
        observer.write_text('#!'+sys.executable+'\nimport os,pathlib,sys\npathlib.Path(os.environ["COMPANION_ATTEMPTS"]).open("ab").write(b"attempt\\n")\nsys.exit(97)\n')
        observer.chmod(0o755)
        supervisor=self.work/'expect-status.py'
        supervisor.write_text('import json,pathlib,subprocess,sys\np=subprocess.run(sys.argv[3:])\npathlib.Path(sys.argv[1]).write_text(json.dumps({"returncode":p.returncode}))\nsys.exit(0 if (p.returncode>0 if sys.argv[2]=="refuse" else p.returncode==int(sys.argv[2])) else 1)\n')
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            for case in self.cases:
                if case['name']=='ordinary':continue
                for option in ([],['--allow-temporary-files']):
                    label=compiler+'-'+case['name']+('-optin' if option else '-default')
                    output=self.work/(label+'.output');output.write_bytes(b'unchanged output sentinel')
                    attempts=self.work/(label+'.attempts');status=self.work/(label+'.return.json')
                    env=dict(case['env'],CC=str(observer),NANO_CC=str(observer),COMPANION_ATTEMPTS=str(attempts))
                    self.command(label+'-refusal',[sys.executable,supervisor,status,'refuse',ROOT/'bin'/compiler,case['entry'],'-o',output,*option],timeout=180,extra=env)
                    self.assertGreater(json.loads(status.read_text())['returncode'],0)
                    self.assertEqual(output.read_bytes(),b'unchanged output sentinel');self.assertFalse(attempts.exists())
            ordinary=next(c for c in self.cases if c['name']=='ordinary')
            for option in ([],['--allow-temporary-files']):
                label=compiler+'-ordinary'+('-optin' if option else '-default');exe=self.work/label
                self.command(label+'-build',[ROOT/'bin'/compiler,ordinary['entry'],'-o',exe,*option],timeout=900)
                self.command(label+'-run',[sys.executable,supervisor,self.work/(label+'.return.json'),'37',exe],timeout=90)


    def test_provider_preparation_identity_and_cleanup(self):
        source=self.work/'provider-probe.nano';source.write_text(provider_report_source(ROOT))
        expected,inputs=selected(source)
        (self.work/'provider-expected-selection.json').write_text(json.dumps(dict(names=expected,inputs=inputs),indent=2)+'\n')
        repo,cases,alias=file_native_provider_corpus.setup(ROOT,self.work)
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/(compiler+'-provider-probe');shadow=self.work/(compiler+'-provider-shadows.json')
            args=[ROOT/'bin'/compiler,source,'-o',exe,'--keep-c']
            if compiler=='nanoc_c':args+=['--llm-shadow-json',shadow,'--verbose']
            out,err=self.command(compiler+'-provider-build',args,timeout=1800,extra={'NANO_SHADOW_TRACE':'1'})
            if compiler=='nanoc_c':
                report=json.loads(shadow.read_text());self.assertTrue(report['completed'] and report['success']);self.assertEqual(report['failures'],[]);self.assertEqual(report['test_count'],len(expected))
                raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
            else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            actual=[re.sub(r'^__nano_module_+[0-9]+_','',n.decode()) for n in raw]
            self.assertCountEqual(actual,expected)
            (self.work/(compiler+'-provider-selection.json')).write_text(json.dumps(dict(raw=[n.decode() for n in raw],actual=actual,expected=expected),indent=2)+'\n')
            for case in cases:
                out,_=self.command(compiler+'-provider-'+case['name'],[exe,case['manifest'],'-o',ROOT],timeout=300,extra={'COMPANION_PROVIDER_ROOT':str(repo),'NANO_CFLAGS':''})
                statuses=re.findall(rb'^PROVIDER_STATUS (\d+)$',out,re.M);self.assertEqual(len(statuses),1);self.assertEqual(int(statuses[0])==0,case['ok'])
                paths=re.findall(rb'^OBJECT (.+)$',out,re.M);self.assertEqual(len(paths),case['objects']);self.assertEqual(len(paths),len(set(paths)))
                for path in paths:self.assertFalse(Path(os.fsdecode(path)).exists())
            case=cases[0]
            out,_=self.command(compiler+'-runtime-alias',[exe,case['manifest'],'-o',alias],extra={'COMPANION_PROVIDER_ROOT':str(repo),'NANO_CFLAGS':''})
            runtime=re.findall(rb'^RUNTIME (.+)$',out,re.M);self.assertEqual(len(runtime),53);self.assertEqual(len(set(runtime)),53)
            self.assertEqual(runtime.count(os.fsencode((ROOT/'src/runtime/list_ASTFloat.c').resolve())),1)


    def test_standalone_concurrent_native_invocations(self):
        # C-seed builds each probe above. These actual native drivers own the
        # changed provider preparer; the C-seed module builder remains separate.
        for compiler in ('nanoc_stage1','nanoc_stage2'):
            self.command(compiler+'-concurrent-providers',[sys.executable,'-m','tests.file_native_provider_invocations',ROOT,self.work/(compiler+'-concurrent'),ROOT/'bin'/compiler],timeout=1100)

    def test_public_provider_owners_and_wrappers(self):
        file_cseed_provider_owners.run(self, ROOT, selected)

    def test_native_width_and_allocation(self):
        for suffix,defines in (('assertions',[]),('release',['-DNDEBUG'])):
            exe=self.work/('native-width-allocation-'+suffix)
            self.command('native-width-build-'+suffix, [*self.cc,*self.flags,*defines,
                ROOT/'tests/test_dyn_array_allocation.c',ROOT/'src/runtime/gc.c',ROOT/'src/runtime/gc_struct.c',*self.links,'-o',exe])
            out,_=self.command('native-width-run-'+suffix,[exe],timeout=180)
            self.assertIn(b'I passed native array allocation boundary tests.',out)

    def test_paired_large_record_arrays(self):
        declarations=[];body=[]
        for fields in (61,63):
            name='Wide'+str(fields);variable='items'+str(fields);copy='copied'+str(fields)
            declarations.append('struct '+name+' { '+', '.join('field'+str(i)+': int' for i in range(fields))+' }')
            value=name+' { '+', '.join('field'+str(i)+': '+str(i+1) for i in range(fields))+' }'
            replacement=name+' { '+', '.join('field'+str(i)+': '+str(1000+i) for i in range(fields))+' }'
            body += ['let mut '+variable+': array<'+name+'> = []',
                     'set '+variable+' (array_push '+variable+' '+value+')',
                     'let mut index'+str(fields)+': int = 0',
                     'while (< index'+str(fields)+' 20) { set '+variable+' (array_push '+variable+' (at '+variable+' 0)) set index'+str(fields)+' (+ index'+str(fields)+' 1) }',
                     'let '+copy+': array<'+name+'> = (array_slice '+variable+' 0 21)',
                     '(array_set '+copy+' 0 '+replacement+')',
                     'assert (== (array_length '+variable+') 21)',
                     'assert (== (at '+variable+' 20).field'+str(fields-1)+' '+str(fields)+')',
                     'assert (== (at '+variable+' 0).field0 1)',
                     'assert (== (at '+copy+' 0).field0 1000)']
        source=self.work/'large-records.nano'
        source.write_text('\n'.join(declarations)+'\nfn main() -> int {\n'+ '\n'.join(body)+'\nreturn 0\n}\nshadow main { assert (== (main) 0) }\n')
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/(compiler+'-large-records');shadow=self.work/(compiler+'-large-records.json')
            args=[ROOT/'bin'/compiler,source,'-o',exe,'--keep-c']
            if compiler=='nanoc_c':args+=['--llm-shadow-json',shadow,'--verbose']
            out,err=self.command(compiler+'-large-records-build',args,timeout=600,extra={'NANO_SHADOW_TRACE':'1'})
            if compiler=='nanoc_c':
                result=json.loads(shadow.read_text());self.assertTrue(result['completed'] and result['success'])
                self.assertEqual(result['test_count'],1);self.assertEqual(result['failures'],[])
                names=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
            else:names=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            self.assertEqual(names,[b'main'])
            self.command(compiler+'-large-records-run',[exe])

if __name__=='__main__':unittest.main()
