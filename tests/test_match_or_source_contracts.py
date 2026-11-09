"""I execute grouped union alternatives through actual VM and sanitized native routes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests import test_bstring_native_contracts as native
from tests import test_bstring_source_vm as source

ROOT = source.ROOT


class OrPatternContracts:
    command = native.BstringNativeContracts.command
    exercise = native.BstringNativeContracts.exercise

    def test_unchanged_complete_original_fixture(self):
        path = ROOT/'tests/unit/test_match_orpatterns.nano'
        self.exercise(path.read_text(), source_path=path)

    def test_group_guard_once_order_and_duplicate_alternatives(self):
        self.exercise('''union Choice{First{value:int} Second{} Third{}}
let mut trace:int=0
let mut reads:int=0
fn select(value:Choice)->Choice{set reads (+ reads 1) set trace (+ (* trace 10) 1) return value}
shadow select{set reads 0 set trace 0 let value:Choice=(select Choice.Second{}) assert (== reads 1) assert (== trace 1)}
fn gate(value:bool)->bool{set trace (+ (* trace 10) 2) return value}
shadow gate{set trace 0 assert (not (gate false)) assert (== trace 2)}
fn body(value:int)->int{set trace (+ (* trace 10) 3) return value}
shadow body{set trace 0 assert (== (body 7) 7) assert (== trace 3)}
fn choose(value:Choice)->int{
 return match (select value){
  First | First | Second if (gate false) => (body 100)
  First | Second if (gate true) => (body 7)
  Third(t) => (body 9)
  _ => (body 19)
 }
}
shadow choose{
 set reads 0 set trace 0
 assert (== (choose Choice.First{value:1}) 7)
 assert (== reads 1) assert (== trace 1223)
}
fn main()->int{
 set reads 0 set trace 0
 assert (== (choose Choice.First{value:1}) 7)
 assert (== reads 1) assert (== trace 1223)
 set reads 0 set trace 0
 assert (== (choose Choice.Second{}) 7)
 assert (== reads 1) assert (== trace 1223)
 set reads 0 set trace 0
 assert (== (choose Choice.Third{}) 9)
 assert (== reads 1) assert (== trace 13)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_underscore_member_is_exact_variant_and_expression_results(self):
        self.exercise('''union Choice<T>{First{} _{} Other{}}
fn choose(value:Choice<int>)->int{return match value{First | _ => 7 _ => 9}}
shadow choose{assert (== (choose Choice<int>.Other{}) 9) assert (== (choose Choice<int>._{}) 7)}
fn text(value:Choice<int>)->string{return match value{First | _ => "group" _ => "fallback"}}
shadow text{assert (== (text Choice<int>.First{}) "group") assert (== (text Choice<int>.Other{}) "fallback")}
fn main()->int{
 assert (== (choose Choice<int>.First{}) 7)
 assert (== (choose Choice<int>._{}) 7)
 assert (== (choose Choice<int>.Other{}) 9)
 assert (== (text Choice<int>._{}) "group")
 assert (== (text Choice<int>.Other{}) "fallback")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def refuse(self, label, text):
        output = ROOT/'.test_output'/self.artifact_directory
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=label+'-', dir=output))
        path, module = work/'main.nano', work/'program.nvm'
        path.write_text(text+'\n')
        previous = b'I retain my prior grouped match program.\n'
        module.write_bytes(previous)
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, self.compiler]}
        try:
            result = self.command(work, 'refuse', [self.compiler, path, '--emit-nvm', '-o', module],
                                  build=True, accepted=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stdout+result.stderr, r'(?i)(parse|type|expected|error|match|coverage|guard|unknown)')
            self.assertEqual(module.read_bytes(), previous)
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, self.compiler]}
            (work/'inputs.json').write_text(json.dumps({'scope': self.scope, 'before': before,
                'after': after, 'prior_output_preserved': module.read_bytes() == previous}, indent=2)+'\n')
            self.assertEqual(before, after)
        return result

    def test_malformed_types_and_conditional_coverage_preserve_output(self):
        for label, arms in {
            'missing-alternative': 'First | => 1 _ => 2',
            'integer-alternative': 'First | 1 => 1 _ => 2',
            'first-binding': 'First(x) | Second => 1 _ => 2',
            'second-binding': 'First | Second(x) => 1 _ => 2',
            'wildcard-head': '_ | First => 1',
            'wrong-guard': 'First | Second if 7 => 1 _ => 2',
            'conditional-missing': 'First | Second if false => 1',
            'duplicate-missing': 'First | First => 1',
            'wrong-result': 'First | Second => true',
        }.items():
            with self.subTest(case=label):
                self.refuse(label, 'union Choice{First{} Second{}} fn main()->int{return match Choice.First{} {'+arms+'}}\nshadow main{assert true}')


class CSeedOrPatternContracts(OrPatternContracts, unittest.TestCase):
    compiler = ROOT/'bin/nano_virt'
    artifact_directory = 'match-or-cseed-contracts'
    scope = 'I execute actual C-created grouped union match modules and sanitized native products.'


class SourceOrPatternContracts(OrPatternContracts, unittest.TestCase):
    artifact_directory = 'match-or-source-contracts'
    scope = 'I execute actual source-created grouped union match modules and sanitized native products; fresh stages remain separate.'

    @classmethod
    def setUpClass(cls):
        source.BstringSourceVmContracts.setUpClass()
        cls.compiler = source.BstringSourceVmContracts.compiler

    def test_existing_exact_source_domain_refusals(self):
        for label, arms in {
            'unknown-variant': 'First | Missing => 1 _ => 2',
            'underscore-not-wildcard': 'First | _ => 1 _ => 2',
        }.items():
            with self.subTest(case=label):
                self.refuse(label, 'union Choice{First{} Second{}} fn main()->int{return match Choice.First{} {'+arms+'}}\nshadow main{assert true}')

    def test_grouped_syntax_does_not_admit_discarded_resource_payloads(self):
        result = self.refuse('resource-payload', '''resource struct Token{fd:int}
union Owned{First{token:Token} Second{}}
fn inspect(value:Owned)->int{return match value{First | Second => 1}}
shadow inspect{assert true}
fn main()->int{
 let owned:Owned=Owned.First{token:Token{fd:1}}
 return (inspect owned)
}
shadow main{assert true}
''')
        self.assertIn('[E0036] I cannot verify ownership:', result.stdout+result.stderr)
