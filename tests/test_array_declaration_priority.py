"""I keep array operation declarations ahead of builtin lowering."""
import unittest
from tests import test_array_builtin_aliases as aliases


class ArrayDeclarationPriority(unittest.TestCase):
    compile = aliases.ArrayBuiltinAliases.compile
    execute_module = aliases.ArrayBuiltinAliases.execute_module
    check = aliases.ArrayBuiltinAliases.check

    def test_module_declarations(self):
        names = ('map', 'filter', 'reduce', 'array_map', 'array_filter', 'array_fold')
        library = '\n'.join('pub fn '+name+'(n:int)->int{return (+ n 1)} shadow '+name+'{assert true}' for name in names)
        self.check('module "lib.nano" as lib', ''.join('assert (== (lib.'+name+' 6) 7) ' for name in names)+'return 7', files={'lib.nano': library})

    def test_declared_scalar_signature(self):
        for name in ('map', 'filter', 'reduce', 'array_map', 'array_filter', 'array_fold'):
            with self.subTest(name=name):
                self.check('fn '+name+'(n:int)->int{return (+ n 1)} shadow '+name+'{assert true}', 'return ('+name+' 6)')

    def test_declared_map_preserves_input(self):
        for name in ('map', 'array_map'):
            with self.subTest(name=name):
                self.check('fn plus(n:int)->int{return (+ n 1)} shadow plus{assert true} fn '+name+'(values:array<int>, callback:fn(int)->int)->array<int>{return values} shadow '+name+'{assert true}', 'let values:array<int> = ('+name+' [7] plus) return (at values 0)')

    def test_declared_filter_changes_element_type(self):
        for name in ('filter', 'array_filter'):
            with self.subTest(name=name):
                self.check('struct Point{x:int} fn keep(n:int)->bool{return true} shadow keep{assert true} fn '+name+'(values:array<int>, callback:fn(int)->bool)->array<Point>{return [Point{x:7}]} shadow '+name+'{assert true}', 'let values = ('+name+' [1] keep) let point = (at values 0) return point.x')

    def test_declared_fold_preserves_initial(self):
        for name in ('reduce', 'array_fold'):
            with self.subTest(name=name):
                self.check('fn add(a:int,b:int)->int{return (+ a b)} shadow add{assert true} fn '+name+'(values:array<int>, initial:int, callback:fn(int,int)->int)->int{return initial} shadow '+name+'{assert true}', 'return ('+name+' [1] 7 add)')
