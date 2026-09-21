"""I qualify actual corrected split ownership; retained-producer refresh is external."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import tempfile
import unittest
from tests import native_sdk_runner
ROOT=Path(__file__).resolve().parents[1]

class SplitOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-split-',dir=os.environ.get('NANO_SPLIT_REPORT_DIR')))
        print('I retain split controls at',cls.work,flush=True)
        cls.cc=shlex.split(os.environ.get('NANO_SPLIT_CC','cc'))
        cls.flags=shlex.split(os.environ.get('NANO_SPLIT_CFLAGS',''))+['-std=c99','-D_GNU_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror','-g','-O1','-I',str(ROOT/'src')]
        cls.links=shlex.split(os.environ.get('NANO_SPLIT_LDFLAGS',''))
        if os.environ.get('NANO_SPLIT_SANITIZERS')=='1':
            cls.flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
        cls.common=[Path(x).resolve() for x in shlex.split(os.environ['NANO_SPLIT_COMMON_OBJECTS'])]
        cls.eval_object=Path(os.environ['NANO_SPLIT_EVAL_OBJECT']).resolve()
        if not cls.common or any(p.name in ('main.o','eval.o') for p in cls.common):
            raise AssertionError('I require complete ordinary compiler providers except main/eval')
        cls.inputs=cls.common+[cls.eval_object]
        cls.before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.inputs}
        (cls.work/'ordinary-providers-before.json').write_text(json.dumps(cls.before,indent=2)+'\n')
        cls.runtime=[]
        for name in ('gc','dyn_array'):
            obj=cls.work/(name+'.o')
            cls.command(name+'-build',[*cls.cc,*cls.flags,'-c',ROOT/'src/runtime'/(name+'.c'),'-o',obj])
            cls.runtime.append(obj)
        cls.common=[p for p in cls.common if not (p.parent.name=='runtime' and p.name in ('gc.o','dyn_array.o'))]
        (cls.work/'scope.json').write_text(json.dumps({'fresh_instrumented_if_selected':['actual included eval.c in evaluator probe','actual gc.c','actual dyn_array.c','generated native helper/probe'],
            'common_and_native_probe_eval':'ordinary, explicitly hashed',
            'faults':'initial array and each segment GC allocation at the owning call site; array-growth interior allocation is not injected',
            'bootstrap':'external; a retained-producer refresh is not clean C-seed bootstrap'},indent=2)+'\n')
    @classmethod
    def command(cls,name,args,expected=(0,)):
        return native_sdk_runner.run(cls.work,name,args,ROOT,timeout=180,expected=expected)
    @classmethod
    def tearDownClass(cls):
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.inputs}
        (cls.work/'ordinary-providers-after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=cls.before:raise AssertionError('ordinary split providers changed')
    def sweep(self,label,exe):
        reports=[]
        for case in range(8):
            out,_,_=self.command(f'{label}-{case}-normal',[exe,case,0])
            fact=json.loads(out);self.assertEqual(fact['case'],case)
            self.assertEqual(fact['allocations'],fact['segments']+1)
            reports.append(fact)
            for failure in range(1,fact['allocations']+1):
                out,err,_=self.command(f'{label}-{case}-fail-{failure}',[exe,case,failure],expected=(-signal.SIGABRT,))
                self.assertEqual(out,b'')
                site='array' if failure==1 else 'segment'
                self.assertEqual(err,f'SPLIT_FAULT {failure} {site}\nI cannot allocate a complete split-string result.\n'.encode())
                recovered,_,_=self.command(f'{label}-{case}-recover-{failure}',[exe,case,0])
                self.assertEqual(json.loads(recovered),fact)
        (self.work/(label+'-results.json')).write_text(json.dumps(reports,indent=2)+'\n')
    def test_evaluator_ownership_and_failures(self):
        exe=self.work/'eval-split'
        self.command('eval-build',[*self.cc,*self.flags,ROOT/'tests/str_split_ownership_probe.c',*self.common,*self.runtime,*self.links,'-o',exe])
        self.sweep('eval',exe)
    def test_native_ownership_and_failures(self):
        emitter=self.work/'emit-runtime'
        self.command('emitter-build',[*self.cc,*self.flags,ROOT/'tests/str_split_runtime_emitter.c',*self.common,self.eval_object,*self.runtime,*self.links,'-o',emitter])
        rendered,_,_=self.command('runtime-output',[emitter])
        start=rendered.index(b'static DynArray* nl_str_split(')
        end=rendered.index(b'static const char* nl_str_join(',start)
        header=self.work/'actual-split.h';header.write_bytes(rendered[start:end])
        exe=self.work/'native-split'
        self.command('native-build',[*self.cc,*self.flags,'-DSPLIT_NATIVE_HEADER='+json.dumps(str(header)),ROOT/'tests/str_split_ownership_probe.c',*self.common,self.eval_object,*self.runtime,*self.links,'-o',exe])
        self.sweep('native',exe)

def load_tests(loader, tests, pattern):
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): yield from ids(item)
            else: yield item.id()
    expected={__name__+'.SplitOwnership.'+name for name in ('test_evaluator_ownership_and_failures','test_native_ownership_and_failures')}
    if set(ids(tests))!=expected: raise AssertionError('I require exactly the two split ownership methods')
    return tests
