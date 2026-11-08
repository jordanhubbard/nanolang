"""I retain declared anonymous record contexts through original source routes."""
import hashlib
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class AnonymousRecordContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def test_original_complete_literal_fixture(self):
        path = ROOT/'tests/test_anonymous_struct_literal.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), '✅ All anonymous struct literal tests passed!\n',
                          source_path=path, expected_opcodes=('AGG_PACK', 'AGG_GET'))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_original_complete_argument_fixture(self):
        path = ROOT/'tests/test_anonymous_struct_function_args.nano'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            self.exercise(path.read_text(), '20\n', source_path=path,
                          expected_opcodes=('AGG_PACK', 'AGG_GET'))
        finally:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_nested_and_empty_declared_records(self):
        self.exercise('''struct Point{x:int,y:int}
struct Outer{child:Point,label:string,enabled:bool,weight:float}
struct Empty{}
fn make()->Outer{return {weight:2.5,enabled:true,label:"héllo",child:{y:9,x:7}}}
shadow make{let p:Outer=(make) assert (== p.child.x 7) assert (== p.weight 2.5)}
fn empty()->Empty{return {}}
shadow empty{let values:array<Empty> = [(empty)] assert (== (array_length values) 1)}
fn main()->int{
 let value:Outer=(make)
 assert (== value.child.y 9) assert (== value.label "héllo") assert value.enabled
 let nothing:Empty=(empty)
 let points:array<Point> = [{x:2,y:3},{y:5,x:4}]
 assert (== (at points 0).x 2) assert (== (at points 1).y 5)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_separate_declared_nominal_contexts(self):
        self.exercise('''struct First{value:int}
struct Second{value:int}
fn first()->First{return {value:7}}
shadow first{assert (== (first).value 7)}
fn second()->Second{return {value:9}}
shadow second{assert (== (second).value 9)}
fn take(value:First)->int{return value.value}
shadow take{assert (== (take {value:11}) 11)}
fn main()->int{
 let left:First=(first) let right:Second=(second)
 assert (== left.value 7) assert (== right.value 9)
 assert (== (take {value:13}) 13)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_malformed_fields_refuse_before_replacing_output(self):
        for literal in ('{x:7}', '{x:7,y:9,z:11}', '{x:7,x:9}', '{x:true,y:9}'):
            with self.subTest(literal=literal):
                self.exercise('struct Point{x:int,y:int}\nfn make()->Point{return '+literal+'''}
shadow make{assert true}
fn main()->int{return 0}
shadow main{assert (== (main) 0)}
''', refused=True)
