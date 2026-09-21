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
        positives['result-context']=('''struct Words { items: array<string> }
fn make_parts() -> array<string> { return (str_split "a,b" ",") }
shadow make_parts { assert (== (array_length (make_parts)) 2) }
fn count_parts(parts: array<string>) -> int { return (array_length parts) }
shadow count_parts { assert (== (count_parts (str_split "a,b" ",")) 2) }
fn probe() -> int {
    let inferred = (str_split "a,b" ",")
    let copied: array<string> = inferred
    let mut values: array<string> = copied
    set values (make_parts)
    let callback: fn(array<string>)->int = count_parts
    assert (== (callback values) 2)
    assert (== (count_parts (str_split "a,b" ",")) 2)
    let words: Words = Words { items: (str_split "a,b" ",") }
    assert (== (at words.items 1) "b")
    return 0
}
shadow probe { assert (== (probe) 0) }
'''+main,['make_parts','count_parts','probe','main'])
        positives['declared-array']=('''fn str_split(value: int) -> array<int> { return [value] }
shadow str_split { assert (== (at (str_split 41) 0) 41) }
fn probe() -> int { let values: array<int> = (str_split 41) assert (== (at values 0) 41) return 0 }
shadow probe { assert (== (probe) 0) }
'''+main,['str_split','probe','main'])
        positives['local-array']=('''fn wrap(value: int) -> array<int> { return [value] }
shadow wrap { assert (== (at (wrap 41) 0) 41) }
fn probe() -> int { let str_split: fn(int)->array<int> = wrap let values: array<int> = (str_split 41) assert (== (at values 0) 41) return 0 }
shadow probe { assert (== (probe) 0) }
'''+main,['wrap','probe','main'])
        array_provider=self.work/'array-provider.nano'
        array_provider.write_text('module split_array_provider\npub fn str_split(value: int) -> array<int> { return [value] }\nshadow str_split { assert (== (at (str_split 41) 0) 41) }\n')
        positives['qualified-array']=(f'module {json.dumps(str(array_provider))} as Provider\nfn probe() -> int {{ let values: array<int> = (Provider.str_split 41) assert (== (at values 0) 41) return 0 }}\nshadow probe {{ assert (== (probe) 0) }}\n'+main,['str_split','probe','main'])
        negatives={
            'arity':'let parts: array<string> = (str_split "a")',
            'source-type':'let parts: array<string> = (str_split 1 ",")',
            'delimiter-type':'let parts: array<string> = (str_split "a" 1)',
            'result-type':'let parts: array<int> = (str_split "a" ",")',
            'element-type':'let part: int = (at (str_split "a" ",") 0)'}
        split='(str_split "a,b" ",")'
        safe_main='fn main() -> int { return 0 }\nshadow main { assert true }\n'
        boundary_refusals={
            'alias':'fn main() -> int { let source = '+split+' let bad: array<int> = source return 0 }\nshadow main { assert true }\n',
            'set':'fn main() -> int { let mut values: array<int> = [] set values '+split+' return 0 }\nshadow main { assert true }\n',
            'return':'fn bad() -> array<int> { return '+split+' }\nshadow bad { assert true }\n'+safe_main,
            'direct-call':'fn take(parts: array<int>) -> int { return (array_length parts) }\nshadow take { assert true }\nfn main() -> int { return (take '+split+') }\nshadow main { assert true }\n',
            'indirect-call':'fn take(parts: array<int>) -> int { return (array_length parts) }\nshadow take { assert true }\nfn main() -> int { let callback: fn(array<int>)->int = take return (callback '+split+') }\nshadow main { assert true }\n',
            'record-initializer':'struct Words { items: array<int> }\nfn main() -> int { let words: Words = Words { items: '+split+' } return 0 }\nshadow main { assert true }\n',
            'global':'let words: array<int> = '+split+'\n'+safe_main,
            'nested-array':'fn main() -> int { let words: array<array<int>> = ['+split+'] return 0 }\nshadow main { assert true }\n'}
        # Public borrowed array fields remain outside the scalar-borrow profile.
        for element in ('string','int'):
            boundary_refusals['borrow-profile-'+element]='resource struct Words { items: array<'+element+'> }\nfn replace(view: &mut Words) -> void { set view.items '+split+' }\nshadow replace { assert true }\n'+safe_main
        visibility_owner=self.work/'visibility-provider.nano'
        visibility_owner.write_text('module visibility_provider\n'
            'fn hidden_value() -> int { return 40 }\nshadow hidden_value { assert (== (hidden_value) 40) }\n'
            'pub /* I retain token visibility across comments. */ fn visible_value() -> int { return (+ (hidden_value) 2) }\nshadow visible_value { assert (== (visible_value) 42) }\n'
            'pub fn with_lambda() -> int { let callback: fn(int)->int = fn(value: int) -> int { return (+ value 1) } return (callback 41) }\nshadow with_lambda { assert (== (with_lambda) 42) }\n')
        visibility_names=['hidden_value','visible_value','with_lambda','probe','main']
        owner_literal=json.dumps(str(visibility_owner))
        for label,imports,call in (
            ('qualified',f'module {owner_literal} as Visible\n','(Visible.visible_value)'),
            ('selective',f'from {owner_literal} import visible_value as chosen\n','(chosen)'),
            ('wildcard',f'from {owner_literal} import *\n','(visible_value)'),
            ('lambda',f'module {owner_literal} as Visible\n','(Visible.with_lambda)')):
            positives['visibility-'+label]=(imports+'fn probe() -> int { assert (== '+call+' 42) return 0 }\nshadow probe { assert (== (probe) 0) }\n'+main,visibility_names)
        public_extern=self.work/'public-extern.nano'
        public_extern.write_text('module public_extern\npub extern fn get_argc() -> int\n')
        positives['visibility-public-extern']=(f'module {json.dumps(str(public_extern))} as External\nfn probe() -> int {{ assert (>= (External.get_argc) 0) return 0 }}\nshadow probe {{ assert (== (probe) 0) }}\n'+main,['probe','main'])
        private_extern=self.work/'private-extern.nano'
        private_extern.write_text('module private_extern\nextern fn get_argc() -> int\n')
        private_split=self.work/'private-split-visibility.nano'
        private_split.write_text('module private_split_visibility\nfn str_split(value: int) -> int { return value }\nshadow str_split { assert true }\n')
        positives['visibility-namespace-keeps-builtin']=(f'module {json.dumps(str(private_split))} as PrivateOwner\n'
            'fn probe() -> int { let values: array<string> = (str_split "a,b" ",") assert (== (array_length values) 2) assert (== (at values 0) "a") assert (== (at values 1) "b") return 0 }\n'
            'shadow probe { assert (== (probe) 0) }\n'+main,['str_split','probe','main'])
        positives['visibility-wildcard-keeps-builtin']=(f'from {json.dumps(str(private_split))} import *\n'
            'fn probe() -> int { let values: array<string> = (str_split "a,b" ",") assert (== (array_length values) 2) assert (== (at values 0) "a") assert (== (at values 1) "b") return 0 }\n'
            'shadow probe { assert (== (probe) 0) }\n'+main,['str_split','probe','main'])
        visibility_refusals={
            'qualified':f'module {owner_literal} as Visible\nfn main() -> int {{ return (Visible.hidden_value) }}\nshadow main {{ assert true }}\n',
            'selective':f'from {owner_literal} import hidden_value as chosen\nfn main() -> int {{ return (chosen) }}\nshadow main {{ assert true }}\n',
            'qualified-value':f'module {owner_literal} as Visible\nfn main() -> int {{ let callback: fn()->int = Visible.hidden_value return (callback) }}\nshadow main {{ assert true }}\n',
            'wildcard':f'from {owner_literal} import *\nfn main() -> int {{ return (hidden_value) }}\nshadow main {{ assert true }}\n',
            'private-extern':f'module {json.dumps(str(private_extern))} as External\nfn main() -> int {{ return (External.get_argc) }}\nshadow main {{ assert true }}\n',
            'private-extern-unqualified':f'import {json.dumps(str(private_extern))}\nfn main() -> int {{ return (get_argc) }}\nshadow main {{ assert true }}\n',
            'private-builtin-value':f'from {json.dumps(str(private_split))} import str_split\nfn main() -> int {{ let callback: fn(int)->int = str_split return (callback 41) }}\nshadow main {{ assert true }}\n'}
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
            for label,source in visibility_refusals.items():
                name=role+'-visibility-refuse-'+label
                path=self.work/(name+'.nano');path.write_text(source)
                exe=self.work/name;exe.write_bytes(b'visibility output sentinel\n')
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',[row['path'],path,'-o',exe],ROOT,
                    {'NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},expected=(1,),timeout=300)
                self.assertEqual(exe.read_bytes(),b'visibility output sentinel\n')
                self.assertNotIn(b'C compilation failed',out+err)
                self.assertNotIn(b'Failed to parse',out+err)
            for label,source in boundary_refusals.items():
                name=role+'-boundary-refuse-'+label
                path=self.work/(name+'.nano');path.write_text(source)
                exe=self.work/name;exe.write_bytes(b'boundary output sentinel\n')
                out,err,_=native_sdk_runner.run(self.work,name+'-compile',[row['path'],path,'-o',exe],ROOT,
                    {'NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},expected=(1,),timeout=300)
                self.assertEqual(exe.read_bytes(),b'boundary output sentinel\n')
                if label.startswith('borrow-profile-'):
                    self.assertIn(b'borrow',out+err)
                else:
                    self.assertNotIn(b'C compilation failed',out+err)
                    self.assertNotIn(b'Failed to parse',out+err)

    def test_actual_parser_capture_limits(self):
        # I exercise the existing File cap and complete ordinary capture separately.
        parser_path=json.dumps(str(ROOT/'src_nano/parser.nano'))
        source='import '+parser_path+'\n'+'''fn main() -> int {
    let mut source: string = ""
    let mut index: int = 0
    while (< index 4097) {
        set source (+ source "pub fn value() -> int { return 1 }\\n")
        set index (+ index 1)
    }
    let line_size: int = (str_length "pub fn value() -> int { return 1 }\\n")
    let source_size: int = (str_length source)
    let boundary_source: string = (str_substring source 0 (- source_size line_size))
    let boundary_tokens: List<LexerToken> = (tokenize_string boundary_source "boundary.nano" (list_CompilerDiagnostic_new))
    let boundary: ParsedDeclarations = (parse_program_declarations boundary_tokens (list_LexerToken_length boundary_tokens) "boundary.nano")
    assert (not (parser_has_error boundary.parser))
    assert (not boundary.truncated)
    assert (== (array_length boundary.definitions) 4096)
    let tokens: List<LexerToken> = (tokenize_string source "large.nano" (list_CompilerDiagnostic_new))
    let count: int = (list_LexerToken_length tokens)
    let limited: ParsedDeclarations = (parse_program_declarations tokens count "large.nano")
    assert (not (parser_has_error limited.parser))
    assert limited.truncated
    assert (== (array_length limited.definitions) 0)
    let complete: ParsedDeclarations = (parse_program_complete_declarations tokens count "large.nano")
    assert (not (parser_has_error complete.parser))
    assert (not complete.truncated)
    assert (== (array_length complete.definitions) 4097)
    assert (== (at complete.definitions 4096).function_index 4096)
    let ordinary: Parser = (parse_program tokens count "large.nano")
    assert (not (parser_has_error ordinary))
    assert (== ordinary.functions_count 4097)
    return 0
}
shadow main { assert true }
'''
        # Imported APIs have explicit origins; I do not rely on a transitive import.
        source=('import '+json.dumps(str(ROOT/'src_nano/compiler/lexer.nano'))+'\n'
                'import '+json.dumps(str(ROOT/'src_nano/compiler/diagnostics.nano'))+'\n'+source)
        path=self.work/'capture-limits.nano';path.write_text(source)
        baseline=None
        for role in ('cseed','refresh1','refresh2'):
            row=self.producers[role];name=role+'-capture-limits'
            exe=self.work/name;temp=self.work/(name+'-tmp');temp.mkdir()
            shadow=self.work/(name+'-shadows.json')
            argv=[row['path'],path,'-o',exe,'--keep-c']
            if role=='cseed':argv+=['--verbose','--llm-shadow-json',shadow]
            out,err,_=native_sdk_runner.run(self.work,name+'-compile',argv,ROOT,
                {'TMPDIR':temp,'NANO_SHADOW_TRACE':'1','NANO_SHADOW_TIMING':None,'NANO_SHADOW_TIMEOUT_SECONDS':None},timeout=300)
            if role=='cseed':
                report=json.loads(shadow.read_text())
                self.assertTrue(report['completed'] and report['success'])
                self.assertEqual(report['failures'],[])
                raw=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
                self.assertEqual(report['test_count'],len(raw))
            else:raw=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            actual=Counter(re.sub(r'^__nano_module_+[0-9]+_','',x.decode()) for x in raw)
            for required in ('main','parse_definition_result','definition_parse_result','parse_program_complete_declarations','parse_program_declarations','parse_program_impl'):
                self.assertEqual(actual[required],1)
            if baseline is None:baseline=actual
            self.assertEqual(actual,baseline)
            (self.work/(name+'-selection.json')).write_text(json.dumps(dict(actual),sort_keys=True,indent=2)+'\n')
            result,_,_=native_sdk_runner.run(self.work,name+'-run',[exe],ROOT,timeout=300)
            self.assertEqual(result,b'')


def load_tests(loader, tests, pattern):
    suite=loader.loadTestsFromTestCase(SplitPaired)
    if suite.countTestCases()!=2:
        raise AssertionError('I require exactly the original paired corpus and parser capture methods')
    return suite
