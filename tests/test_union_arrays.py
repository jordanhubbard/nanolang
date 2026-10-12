"""I preserve union-array payloads across construction, mutation and native execution."""
import os
from pathlib import Path
import tempfile
import unittest
from tests import test_selfhost_effects as effects

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = os.environ.get('NANOLANG_UNION_ARRAY_COMPILERS', str(ROOT/'bin/nano_virt')).split(os.pathsep)
CASES = {'mutation': ('union Box<T>{Some{value:T}}',
              'let mut xs:array<Box<string>> = '
              '[Box<string>.Some{value:"first"}] let saved:Box<string> = (at '
              'xs 0) set xs (array_push xs Box<string>.Some{value:"second"}) '
              '(array_set xs 0 Box<string>.Some{value:"changed"}) match '
              'saved{Some(p)=>{assert (== p.value "first")}} match (at xs '
              '0){Some(p)=>{assert (== p.value "changed")}} match (at xs '
              '1){Some(p)=>{assert (== p.value "second")}} return 7'),
 'array_payload': ('union Box<T>{Some{value:T}}',
                   'let xs:array<Box<array<int>>> = '
                   '[Box<array<int>>.Some{value:[3,4]}] match (at xs '
                   '0){Some(p)=>{assert (== (at p.value 1) 4)}} return 7'),
 'mixed_variants': ('union Value<T>{Number{n:T},Text{s:string}}',
                    'let xs:array<Value<int>> = '
                    '[Value<int>.Number{n:7},Value<int>.Text{s:"hello"}] match '
                    '(at xs 0){Number(p)=>{assert (== p.n 7)} Text(p)=>{assert '
                    'false}} match (at xs 1){Number(p)=>{assert false} '
                    'Text(p)=>{assert (== p.s "hello")}} return 7'),
 'generic_alias': ('union Box<T>{Some{value:T}}',
                   'let xs:array<Box<string>> = '
                   '[Box<string>.Some{value:"retained"}] let '
                   'ys:array<Box<string>> = xs match (at ys '
                   '0){Some(p)=>{assert (== p.value "retained")}} return 7'),
 'filled': ('union Box<T>{Some{value:T}}',
            'let xs:array<Box<int>> = (array_new 3 Box<int>.Some{value:9}) '
            'match (at xs 2){Some(p)=>{assert (== p.value 9)}} return 7')}

CASES['ordinary_union'] = ('union Value{Number{n:int},Text{s:string}}',
    'let xs:array<Value> = [Value.Number{n:7},Value.Text{s:"hello"}] match (at xs 0){Number(p)=>{assert (== p.n 7)} Text(p)=>{assert false}} match (at xs 1){Number(p)=>{assert false} Text(p)=>{assert (== p.s "hello")}} return 7')
CASES['effect_boundary'] = ('union Box<T>{Some{value:T}} effect E{ask:array<Box<int>>->array<Box<int>>}',
    'let xs:array<Box<int>> = handle {perform E.ask([Box<int>.Some{value:7}])} with {ask xs->{xs}} match (at xs 0){Some(p)=>{return p.value}} return 0')

CASES['mixed_variant_mutation'] = ('union Value<T>{Number{n:T},Text{s:string}}',
    'let mut xs:array<Value<int>> = [Value<int>.Number{n:7}] set xs (array_push xs Value<int>.Text{s:"pushed"}) (array_set xs 0 Value<int>.Text{s:"replaced"}) match (at xs 0){Number(p)=>{assert false} Text(p)=>{assert (== p.s "replaced")}} match (at xs 1){Number(p)=>{assert false} Text(p)=>{assert (== p.s "pushed")}} return 7')

CASES['empty_then_append'] = ('union Box<T>{Some{value:T}}',
    'let mut xs:array<Box<int>> = [] set xs (array_push xs Box<int>.Some{value:7}) match (at xs 0){Some(p)=>{return p.value}} return 0')
CASES['nested_arrays'] = ('union Box<T>{Some{value:T}}',
    'let xs:array<array<Box<int>>> = [[Box<int>.Some{value:7}]] let inner:array<Box<int>> = (at xs 0) match (at inner 0){Some(p)=>{return p.value}} return 0')

class UnionArrays(unittest.TestCase):
    compile = effects.SelfhostEffects.compile
    execute_module = effects.SelfhostEffects.execute_module

    def test_construction_and_retained_payloads(self):
        for driver in DRIVERS:
            for name, (declaration, body) in CASES.items():
                with self.subTest(case=name, driver=driver), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                    source.write_text(declaration+'\nfn main()->int{'+body+'}\nshadow main{assert (== (main) 7)}\n')
                    result = self.compile(driver, source, output)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.execute_module(output, 7)

    def test_mismatches_preserve_output(self):
        cases = [
            'union Box<T>{Some{value:T}} fn main()->int{let xs:array<Box<int>> = [Box<bool>.Some{value:true}] return 0}',
            'union A{Some{value:int}} union B{Some{value:int}} fn main()->int{let xs:array<A> = [B.Some{value:7}] return 0}',
            'union A{Some{value:int}} union B{Some{value:int}} fn main()->int{let ys:array<B> = [B.Some{value:7}] let xs:array<A> = ys return 0}',
            'union A{Some{value:int}} union B{Some{value:int}} fn main()->int{let xs:array<array<A>> = [[B.Some{value:7}]] return 0}',
        ]
        for driver in DRIVERS:
            for index, source_text in enumerate(cases):
                with self.subTest(case=index, driver=driver), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                    source.write_text(source_text+'\nshadow main{assert (== (main) 0)}\n')
                    output.write_bytes(b'prior artifact')
                    result = self.compile(driver, source, output)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')
