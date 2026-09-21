"""I compare corrected C-seed and two explicitly attributed refreshed Nano producers."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from tests import native_sdk_runner
ROOT=Path(__file__).resolve().parents[1]

class SplitPaired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-split-paired-',dir=os.environ.get('NANO_SPLIT_REPORT_DIR')))
        print('I retain paired split controls at',cls.work,flush=True)
        cls.provenance=json.loads(Path(os.environ['NANO_SPLIT_PRODUCER_MANIFEST']).read_text())
        cls.producers=cls.provenance['producers']
        if set(cls.producers)!={'cseed','refresh1','refresh2'}:
            raise AssertionError('I require exactly the three attributed producer roles')
        cls.before={}
        for name,row in cls.producers.items():
            p=Path(row['path']).resolve();digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest!=row['sha256']:raise AssertionError(('producer hash',name))
            cls.before[str(p)]=digest
        (cls.work/'producer-provenance.json').write_text(json.dumps(cls.provenance,indent=2)+'\n')
    @classmethod
    def tearDownClass(cls):
        after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cls.before}
        (cls.work/'producer-after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=cls.before:raise AssertionError('split producers changed')
    def test_exact_results_typing_and_call_ownership(self):
        main='fn main() -> int { return (probe) }\nshadow main { assert (== (main) 0) }\n'
        positives={
            'contents':('''fn probe() -> int {
    let parts: array<string> = (str_split "\\né\\n\\nend\\n" "\\n")
    assert (== (array_length parts) 5)
    assert (== (at parts 0) "")
    assert (== (at parts 1) "é")
    assert (== (at parts 2) "")
    assert (== (at parts 3) "end")
    assert (== (at parts 4) "")
    let inferred = (str_split "a::b::::" "::")
    assert (== (array_length inferred) 4)
    assert (== (at inferred 0) "a")
    assert (== (at inferred 1) "b")
    assert (== (at inferred 2) "")
    assert (== (at inferred 3) "")
    let fragment: string = (str_substring "é" 0 1)
    let partial: array<string> = (str_split fragment ":")
    assert (== (array_length partial) 1)
    assert (== (at partial 0) fragment)
    let empty: array<string> = (str_split "" "\\n")
    assert (== (array_length empty) 1)
    assert (== (at empty 0) "")
    let bytes: array<string> = (str_split "abc" "")
    assert (== (array_length bytes) 3)
    assert (== (at bytes 0) "a")
    assert (== (at bytes 1) "b")
    assert (== (at bytes 2) "c")
    let crlf: array<string> = (str_split "a\\r\\nb" "\\n")
    assert (== (at crlf 0) "a\\r")
    assert (== (at crlf 1) "b")
    return 0
}
shadow probe { assert (== (probe) 0) }
'''+main,['probe','main']),
            'declared':('''fn str_split(value: int) -> int { return (+ value 1) }
shadow str_split { assert (== (str_split 41) 42) }
fn probe() -> int { assert (== (str_split 41) 42) return 0 }
shadow probe { assert (== (probe) 0) }
'''+main,['str_split','probe','main']),
            'local':('''fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 41) 42) }
fn probe() -> int {
    let str_split: fn(int)->int = increment
    assert (== (str_split 41) 42)
    return 0
}
shadow probe { assert (== (probe) 0) }
'''+main,['increment','probe','main'])}
        module=self.work/'split-provider.nano'
        module.write_text('module split_provider\npub fn str_split(value: int) -> int { return (+ value 1) }\nshadow str_split { assert (== (str_split 41) 42) }\n')
        positives['qualified']=(f'module {json.dumps(str(module))} as Provider\nfn probe() -> int {{ assert (== (Provider.str_split 41) 42) return 0 }}\nshadow probe {{ assert (== (probe) 0) }}\n'+main,['str_split','probe','main'])
        negatives={
            'arity':'let parts: array<string> = (str_split "a")',
            'source-type':'let parts: array<string> = (str_split 1 ",")',
            'delimiter-type':'let parts: array<string> = (str_split "a" 1)',
            'result-type':'let parts: array<int> = (str_split "a" ",")',
            'element-type':'let part: int = (at (str_split "a" ",") 0)'}
        for role,row in self.producers.items():
            for label,(source,names) in positives.items():
                name=role+'-'+label;path=self.work/(name+'.nano');path.write_text(source)
                exe=self.work/name;temp=self.work/(name+'-tmp');temp.mkdir()
                shadow=self.work/(name+'-shadows.json')
                argv=[row['path'],path,'-o',exe,'--keep-c']
                if role=='cseed':argv+=['--verbose','--llm-shadow-json',shadow]
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',argv,ROOT,
                    {'TMPDIR':temp,'NANO_SHADOW_TRACE':'1','NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},timeout=300)
                if role=='cseed':
                    report=json.loads(shadow.read_text())
                    self.assertTrue(report['completed'] and report['success'])
                    self.assertEqual(report['test_count'],len(names));self.assertEqual(report['failures'],[])
                    raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
                else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
                actual=[re.sub(r'^__nano_module_+[0-9]+_','',x.decode()) for x in raw]
                (self.work/(name+'-selection.json')).write_text(json.dumps({'expected':names,'actual':actual},indent=2)+'\n')
                self.assertEqual(Counter(actual),Counter(names))
                result,_,_=native_sdk_runner.run(self.work,name+'-run',[exe],ROOT)
                self.assertEqual(result,b'')
            for label,statement in negatives.items():
                name=role+'-refuse-'+label;path=self.work/(name+'.nano')
                path.write_text('fn main() -> int { '+statement+' return 0 }\nshadow main { assert true }\n')
                exe=self.work/name;exe.write_bytes(b'output sentinel\n')
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',[row['path'],path,'-o',exe],ROOT,
                    {'NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},expected=(1,),timeout=300)
                self.assertEqual(exe.read_bytes(),b'output sentinel\n')
                if label in ('arity','source-type','delimiter-type'):
                    self.assertIn(b'I require two strings for str_split.',out+err)
            module_refusals={
                'private':'fn str_split(value: int) -> int { return (+ value 1) }\n',
                'duplicate':'pub fn str_split(value: int) -> int { return (+ value 1) }\npub fn str_split(value: int) -> int { return (+ value 2) }\n',
                'extern-collision':'extern fn str_split(value: int) -> int\npub fn str_split(value: int) -> int { return (+ value 1) }\n'}
            for label,declarations in module_refusals.items():
                name=role+'-module-refuse-'+label
                owner=self.work/(name+'-provider.nano')
                owner.write_text('module split_refusal_provider\n'+declarations+'shadow str_split { assert true }\n')
                path=self.work/(name+'.nano')
                path.write_text(f'module {json.dumps(str(owner))} as Provider\nfn main() -> int {{ return (Provider.str_split 41) }}\nshadow main {{ assert true }}\n')
                exe=self.work/name;exe.write_bytes(b'module output sentinel\n')
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',[row['path'],path,'-o',exe],ROOT,
                    {'NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},expected=(1,),timeout=300)
                self.assertEqual(exe.read_bytes(),b'module output sentinel\n')
                self.assertIn(b'str_split',out+err)
