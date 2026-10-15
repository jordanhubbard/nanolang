"""I execute inherited record fields and ordered spread operands."""
import unittest
from tests import test_array_builtin_aliases as matrix


class RecordSpreads(unittest.TestCase):
    compile = matrix.ArrayBuiltinAliases.compile
    execute_module = matrix.ArrayBuiltinAliases.execute_module
    check = matrix.ArrayBuiltinAliases.check

    def test_inherited_fields_and_unchanged_source(self):
        self.check('struct Point{x:int,y:int,z:int}', 'let p:Point=Point{x:1,y:2,z:3} let q:Point={..p,z:4} assert (== p.z 3) return (+ q.x (+ q.y q.z))')

    def test_source_once_and_override_order(self):
        self.check('struct Point{x:int,y:int} let mut counter:int=0 fn source()->Point{set counter (+ counter 10) return Point{x:1,y:2}} shadow source{assert true} fn next()->int{set counter (+ counter 1) return counter} shadow next{assert true}', 'set counter 0 let q:Point={..(source),y:(next),x:(next)} assert (== counter 12) assert (== q.y 11) assert (== q.x 12) return (+ 6 (- q.x q.y))')

    def test_all_fields_overridden(self):
        self.check('struct Point{x:int,y:int} let mut counter:int=0 fn source()->Point{set counter (+ counter 1) return Point{x:1,y:2}} shadow source{assert true}', 'set counter 0 let q:Point={..(source),y:3,x:4} assert (== counter 1) return (+ q.x q.y)')

    def test_no_overrides(self):
        self.check('struct Point{x:int,y:int}', 'let p:Point=Point{x:3,y:4} let q:Point={..p} return (+ q.x q.y)')

    def test_nested_array_and_callable_fields(self):
        self.check('struct Point{x:int} struct Bundle{point:Point,values:array<int>,callback:fn(int)->int} fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true}', 'let p:Bundle=Bundle{point:Point{x:3},values:[2],callback:plus} let q:Bundle={..p,values:[3]} let callback:fn(int)->int=q.callback let point:Point=q.point return (+ point.x (callback (at q.values 0)))')

    def test_invalid_sources_and_overrides(self):
        for body in ('let p:Point=Point{x:1} let q:Point={..p,x:true} return 7',
                     'let q:Point={..7,x:1} return 7'):
            with self.subTest(body=body):
                self.check('struct Point{x:int}', body, accepted=False)
