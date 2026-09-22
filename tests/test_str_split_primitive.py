"""I qualify the split primitive only; the complete paired SDK corpus stays separate."""
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

class SplitPrimitive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='nano-split-primitive-',dir=os.environ['NANO_SPLIT_REPORT_DIR']))
        manifest=json.loads(Path(os.environ['NANO_SPLIT_PRODUCER_MANIFEST']).read_text())
        cls.producers=manifest['producers']
        expected=set(manifest['selected_roles'])
        if expected not in ({'cseed'},{'refresh1','refresh2'},{'cseed','refresh1','refresh2'}):
            raise AssertionError('I require explicit current-Cseed and/or two retained Nano roles')
        if set(cls.producers)!=expected or not manifest['scope']:
            raise AssertionError('I require exact attributed primitive roles and scope')
        cls.before={}
        for role,row in cls.producers.items():
            p=Path(row['path']);actual=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
            if actual!={key:row[key] for key in actual}:raise AssertionError(role)
            cls.before[str(p)]=actual
        (cls.work/'producer-before.json').write_text(json.dumps(cls.before,indent=2)+'\n')
        (cls.work/'scope.json').write_text(json.dumps(manifest,indent=2)+'\n')
    @classmethod
    def tearDownClass(cls):
        after={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),bytes=Path(p).stat().st_size,mode=Path(p).stat().st_mode&0o7777) for p in cls.before}
        (cls.work/'producer-after.json').write_text(json.dumps(after,indent=2)+'\n')
        if after!=cls.before:raise AssertionError('I changed a retained split producer')
    def test_original_primitive_content_precedence_and_argument_refusals(self):
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
            for label,statement in (
                ('arity','let parts: array<string> = (str_split "a")'),
                ('source-type','let parts: array<string> = (str_split 1 ",")'),
                ('delimiter-type','let parts: array<string> = (str_split "a" 1)')):
                name=role+'-refuse-'+label;path=self.work/(name+'.nano')
                path.write_text('fn main() -> int { '+statement+' return 0 }\nshadow main { assert true }\n')
                exe=self.work/name;exe.write_bytes(b'output sentinel\n')
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',[row['path'],path,'-o',exe],ROOT,
                    {'NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},expected=(1,),timeout=300)
                self.assertEqual(exe.read_bytes(),b'output sentinel\n')
                self.assertIn(b'I require two strings for str_split.',out+err)
