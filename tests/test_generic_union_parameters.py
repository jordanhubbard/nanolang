"""I retain concrete union instances through generic calls and bodies."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_cseed_generic_functions as base

BOX = 'union Box<T> { Some {value:T}, None {} }\n'
UNBOX = 'fn unbox(value:Box<T>,fallback:T)->T{match value{Some(p)=>{return p.value} None(p)=>{return fallback}}}\nshadow unbox {assert (== (unbox Box<int>.Some {value:7} 0) 7)}\n'
IDENTITY = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
MAIN_SHADOW = '\nshadow main {assert (== (main) 0)}\n'


class GenericUnionParameters(unittest.TestCase):
    run_command = base.CseedGenericFunctions.run_command
    execute = base.CseedGenericFunctions.execute

    def test_retained_failures(self):
        directory = base.ROOT/'docs/evidence/generic-union-parameters-20261011'
        for name in ('identity', 'unbox', 'rebox'):
            with self.subTest(case=name):
                self.execute((directory/(name+'.nano')).read_text())

    def test_scalar_instances_and_empty_variants(self):
        self.execute(BOX+UNBOX+IDENTITY+'''fn box(value:T)->Box<T>{return Box.Some {value:value}}
shadow box {assert (== (unbox (box 7) 0) 7)}
fn main()->int{
 assert (== (unbox (identity (box 7)) 0) 7)
 assert (unbox (box true) false)
 assert (== (unbox (box 1.5) 0.0) 1.5)
 assert (== (unbox (box "kept") "") "kept")
 assert (== (unbox (identity Box<string>.None {}) "fallback") "fallback")
 return 0
}'''+MAIN_SHADOW)

    def test_array_and_callable_payloads(self):
        self.execute(BOX+UNBOX+IDENTITY+'''fn plus(x:int)->int{return (+ x 1)}
shadow plus {assert (== (plus 1) 2)}
fn main()->int{
 let source:array<int> = [7]
 let boxed:Box<array<int>> = Box.Some {value:source}
 let values:array<int> = (unbox (identity boxed) [0])
 (array_set values 0 9) assert (== (at source 0) 9)
 let callback:Box<fn(int)->int> = Box.Some {value:plus}
 let selected:fn(int)->int = (unbox (identity callback) plus)
 assert (== (selected 7) 8)
 return 0
}'''+MAIN_SHADOW)

    def test_nested_and_multiple_parameters(self):
        self.execute(BOX+UNBOX+IDENTITY+'''union Choice<T,E> {Left {value:T}, Right {value:E}}
fn second(value:Choice<T,E>,fallback:E)->E{match value{Left(p)=>{return fallback} Right(p)=>{return p.value}}}
shadow second {assert (== (second Choice<int,string>.Right {value:"kept"} "") "kept")}
fn main()->int{
 let inner:Box<int> = Box.Some {value:7}
 let outer:Box<Box<int>> = Box.Some {value:inner}
 let recovered:Box<int> = (unbox (identity outer) Box<int>.None {})
 assert (== (unbox recovered 0) 7)
 assert (== (second Choice<int,string>.Right {value:"kept"} "") "kept")
 assert (second Choice<string,bool>.Left {value:"unused"} true)
 return 0
}'''+MAIN_SHADOW)

    def test_ordinary_union_identity(self):
        self.execute(IDENTITY+'''union Choice {Number {value:int}, Flag {value:bool}}
fn read(value:Choice)->int{match value{Number(p)=>{return p.value} Flag(p)=>{if p.value{return 7} return 0}}}
shadow read {assert (== (read Choice.Number {value:7}) 7)}
fn main()->int{assert (== (read (identity Choice.Number {value:7})) 7) assert (== (read (identity Choice.Flag {value:true})) 7) return 0}'''+MAIN_SHADOW)

    def test_imported_union_and_generic_function(self):
        self.execute('module "box.nano" as b\n'+IDENTITY+'''fn main()->int{
 let value:b.Box<int> = (b.make)
 assert (== (b.unbox (identity value) 0) 7)
 return 0
}'''+MAIN_SHADOW,files={'box.nano':BOX.replace('union','pub union',1)+UNBOX.replace('fn unbox','pub fn unbox',1)+
            'pub fn make()->Box<int>{return Box<int>.Some {value:7}}\nshadow make {assert (== (unbox (make) 0) 7)}\n'})

    def test_mismatched_instances_preserve_output(self):
        prefix=BOX+'union Other<T> {Some {value:T}, None {}}\n'+UNBOX+IDENTITY
        for expression in ('(unbox Box<string>.None {} 7)',
                           '(unbox Other<int>.Some {value:7} 0)',
                           'let wrong:Box<string> = (identity Box<int>.None {})',
                           'let wrong:Box<Box<int>> = Box.Some {value:Box<string>.None {}}',
                           '(choose Box<int>.None {} Box<string>.None {})'):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                source,output=Path(tmp)/'source.nano',Path(tmp)/'output'
                source.write_text(prefix+'fn choose(a:T,b:T)->T{return a}\nshadow choose {assert (== (choose 1 2) 1)}\n'
                    'fn unused()->int{'+expression+' return 0}\nshadow unused {assert true}\n'
                    'fn main()->int{return 0}'+MAIN_SHADOW)
                output.write_bytes(b'prior artifact')
                result=subprocess.run([base.COMPILER,source,*base.MODE,'-o',output],cwd=base.ROOT,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior artifact')
                self.assertNotIn('C compilation failed',result.stdout+result.stderr)

    def test_union_result_boundaries_preserve_output(self):
        bodies = [
            'fn unused()->Box<string>{return (identity Box<int>.None {})}',
            'fn unused()->int{let mut value:Box<string> = Box.None {} set value (identity Box<int>.None {}) return 0}',
            'fn take(value:Box<string>)->int{return 0} shadow take {assert (== (take Box<string>.None {}) 0)} '
            'fn unused()->int{return (take (identity Box<int>.None {}))}',
        ]
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'source.nano', Path(tmp)/'output'
                source.write_text(BOX+IDENTITY+body+'\nshadow unused {assert true}\nfn main()->int{return 0}'+MAIN_SHADOW)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([base.COMPILER, source, *base.MODE, '-o', output], cwd=base.ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')
                self.assertNotIn('C compilation failed', result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
