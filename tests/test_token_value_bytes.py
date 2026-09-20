"""I qualify token metadata only; fresh bootstrap is an explicit prerequisite."""
from pathlib import Path
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import difflib
import tempfile
import unittest
from tests import test_file_source_plan as retained_runner
ROOT=Path(__file__).resolve().parents[1]

class TokenValueBytes(unittest.TestCase):
    command=classmethod(retained_runner.FileSourcePlan.command.__func__)
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-token-bytes-',dir=os.environ.get('NANO_TOKEN_REPORT_DIR')))
        print('I retain token artifacts at',cls.work,flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_TOKEN_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_TOKEN_CFLAGS',''))+['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(ROOT/'src')]
        if os.environ.get('NANO_TOKEN_SANITIZERS','0')=='1':
            cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.objects=[]
        for name in ('lexer','utf8','runtime/list_LexerToken','runtime/list_token','runtime/token_helpers'):
            obj=cls.work/(name.replace('/','-')+'.o')
            cls.command('provider-'+obj.stem,[*cls.cc,*cls.flags,'-c',ROOT/'src'/(name+'.c'),'-o',obj])
            cls.objects.append(obj)
        (cls.work/'providers.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.objects},indent=2)+'\n')
    def test_c_counts_decoder_and_bridges(self):
        exe=self.work/'counts-c'
        self.command('counts-c-build',[*self.cc,*self.flags,ROOT/'tests/test_token_value_bytes.c',*self.objects,'-o',exe])
        out,_=self.command('counts-c-run',[exe])
        self.assertEqual(out,b'0:4:4\n1:1:2\n2:3:8\n3:3:6\n4:3:2\n5:70:3\n6:7:0\n7:8:0\n8:0:0\n')
        adjacent=self.work/'fstrings'
        self.command('fstrings-build',[*self.cc,*self.flags,ROOT/'tests/test_fstring_lexer.c',*self.objects,'-o',adjacent])
        self.command('fstrings-run',[adjacent])
    def test_paired_fresh_compilers(self):
        generated=('src/generated/compiler_schema.h',
            'src_nano/generated/compiler_schema.nano',
            'src_nano/generated/compiler_ast.nano',
            'src_nano/generated/compiler_contracts.nano')
        def isolated(name):
            where=self.work/name
            for directory in ('schema','scripts','src/generated','src_nano/generated'):
                (where/directory).mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/'schema/compiler_schema.json',where/'schema/compiler_schema.json')
            return where
        def compare(name,where):
            rows={};mismatch=[]
            for output in generated:
                actual=(where/output).read_bytes();expected=(ROOT/output).read_bytes()
                rows[output]={'actual_sha256':hashlib.sha256(actual).hexdigest(),
                    'committed_sha256':hashlib.sha256(expected).hexdigest(),
                    'actual_bytes':len(actual),'committed_bytes':len(expected)}
                if actual!=expected:
                    mismatch.append(output)
                    (self.work/(name+'-'+Path(output).name+'.diff')).write_text(''.join(
                        difflib.unified_diff(expected.decode().splitlines(True),actual.decode().splitlines(True),fromfile='committed',tofile=name)))
            (self.work/(name+'-generation.json')).write_text(json.dumps(rows,indent=2)+'\n')
            self.assertEqual(mismatch,[])
        python_root=isolated('python-generator')
        shutil.copyfile(ROOT/'scripts/gen_compiler_schema.py',python_root/'scripts/gen_compiler_schema.py')
        self.command('python-generator-run',[sys.executable,python_root/'scripts/gen_compiler_schema.py'])
        compare('python-generator',python_root)
        # Exact import graph from frozen sources, including duplicate shadow names.
        def selected(source):
            visited=set();names=[];files={}
            def visit(path):
                path=path.resolve()
                if path in visited:return
                visited.add(path)
                text=path.read_text();files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                for name in re.findall(r'^(?:unsafe\s+)?(?:import|from|module)\s+"([^"\n]+)"',text,re.M):
                    choices=[ROOT/name,path.parent/name,ROOT/'modules'/name]
                    target=next((p for p in choices if p.is_file()),None)
                    if target is None:raise AssertionError(('unresolved expected import',path,name))
                    visit(target)
                names.extend(re.findall(r'^shadow\s+([A-Za-z_][A-Za-z_0-9]*)\s*\{',text,re.M))
            visit(source)
            return names,files
        for source in (ROOT/'tests/token_value_bytes.nano',ROOT/'scripts/gen_compiler_schema.nano'):
            expected,inputs=selected(source)
            label=source.stem
            (self.work/(label+'-expected-selection.json')).write_text(json.dumps({'names':expected,'inputs':inputs},indent=2)+'\n')
            for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                exe=self.work/(label+'-'+compiler);shadow=self.work/(label+'-'+compiler+'-shadows.json')
                temp=self.work/(label+'-'+compiler+'-tmp');temp.mkdir()
                args=[ROOT/'bin'/compiler,source,'-o',exe,'--keep-c']
                (self.work/(label+'-'+compiler+'-retention.json')).write_text(json.dumps({'keep_c':True,'TMPDIR':str(temp)},indent=2)+'\n')
                if compiler=='nanoc_c':args+=['--llm-shadow-json',shadow,'--verbose']
                out,err=self.command(label+'-'+compiler+'-build',args,timeout=900,extra={'NANO_SHADOW_TRACE':'1','TMPDIR':str(temp)})
                if compiler=='nanoc_c':
                    report=json.loads(shadow.read_text())
                    self.assertTrue(report['completed'] and report['success'])
                    self.assertEqual(report['test_count'],len(expected));self.assertEqual(report['failures'],[])
                    raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
                else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
                names=[re.sub(r'^__nano_module_+[0-9]+_','',p.decode('ascii')) for p in raw]
                (self.work/(label+'-'+compiler+'-selection.json')).write_text(json.dumps({'raw':[p.decode('ascii') for p in raw],'normalized':names,'expected':expected},indent=2)+'\n')
                self.assertCountEqual(names,expected)
                if label=='token_value_bytes':
                    result,_=self.command(label+'-'+compiler+'-run',[exe])
                    self.assertEqual(result,b'0:4:4\n1:1:2\n2:3:8\n3:3:6\n4:3:2\n5:70:3\n6:7:0\n7:8:0\n8:0:0\n')
                else:
                    where=isolated('schema-'+compiler)
                    # The child changes cwd before exec; the source tree stays immutable.
                    self.command(label+'-'+compiler+'-generate',[sys.executable,'-c',
                        'import os,sys; os.chdir(sys.argv[1]); os.execv(sys.argv[2],[sys.argv[2]])',where,exe],timeout=300)
                    compare(label+'-'+compiler,where)
def load_tests(loader, tests, pattern):
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from ids(item)
            else:yield item.id()
    actual=sorted(ids(tests))
    expected=sorted(__name__+'.TokenValueBytes.'+name for name in (
        'test_c_counts_decoder_and_bridges','test_paired_fresh_compilers'))
    if actual!=expected:raise AssertionError(('exact intended discovery',actual,expected))
    return tests

if __name__=='__main__':unittest.main()
