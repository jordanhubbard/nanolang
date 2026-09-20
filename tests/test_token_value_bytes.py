"""I qualify token metadata only; fresh bootstrap is an explicit prerequisite."""
from pathlib import Path
import hashlib
import json
import os
import re
import shlex
import tempfile
import unittest
from tests.test_file_source_plan import FileSourcePlan
ROOT=Path(__file__).resolve().parents[1]

class TokenValueBytes(unittest.TestCase):
    command=classmethod(FileSourcePlan.command.__func__)
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
                args=[ROOT/'bin'/compiler,source,'-o',exe]
                if compiler=='nanoc_c':args+=['--llm-shadow-json',shadow,'--verbose']
                out,err=self.command(label+'-'+compiler+'-build',args,timeout=900,extra={'NANO_SHADOW_TRACE':'1'})
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
                # I compile the schema generator and run all its selected shadows;
                # I do not run its main over the immutable qualified source tree.
if __name__=='__main__':unittest.main()
