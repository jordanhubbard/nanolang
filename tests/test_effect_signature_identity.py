"""I retain complete effect identities through checking, VM and native execution."""
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from tests import test_selfhost_effects as effects

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = os.environ.get('NANOLANG_EFFECT_SIGNATURE_COMPILERS', str(ROOT/'bin/nano_virt')).split(os.pathsep)
CASES = {'enum_ok': ('enum A{One,Two} effect E{ask:A->A} fn main()->int{let a:A=handle {perform E.ask(A.Two)} with '
             '{ask a->{a}} assert (== a A.Two) return 0}',
             True),
 'enum_wrong_arg': ('enum A{One,Two} enum B{One,Two} effect E{ask:A->A} fn main()->int{let a:A=handle '
                    '{perform E.ask(B.Two)} with {ask a->{a}} return 0}',
                    False),
 'generic_ok': ('union Box<T>{Some{value:T}} effect E{ask:Box<int>->Box<int>} fn main()->int{let a:Box<int> '
                '= handle {perform E.ask(Box<int>.Some{value:7})} with {ask a->{a}} match a '
                '{Some(p)=>{return p.value}} return 0}',
                True),
 'generic_wrong_arg': ('union Box<T>{Some{value:T}} effect E{ask:Box<int>->Box<int>} fn main()->int{let '
                       'a:Box<int> = handle {perform E.ask(Box<bool>.Some{value:true})} with {ask a->{a}} '
                       'return 0}',
                       False),
 'record_ok': ('struct A{x:int} effect E{ask:A->A} fn main()->int{let a:A=handle {perform E.ask(A{x:7})} '
               'with {ask a->{a}} return a.x}',
               True),
 'record_wrong_arg': ('struct A{x:int} struct B{x:int} effect E{ask:A->A} fn main()->int{let a:A=handle '
                      '{perform E.ask(B{x:7})} with {ask a->{a}} return a.x}',
                      False),
 'record_wrong_resume': ('struct A{x:int} struct B{x:int} effect E{ask:A->A} fn main()->int{let a:A=handle '
                         '{perform E.ask(A{x:7})} with {ask a->{B{x:9}}} return a.x}',
                         False)}

CASES.update({'array_ok': ('enum A{One,Two} effect E{ask:array<A>->array<A>} fn main()->int{let a:array<A> = handle '
              '{perform E.ask([A.One])} with {ask a->{a}} assert (== (at a 0) A.One) return 7}',
              True),
 'callback_ok': ('effect E{ask:fn(int)->int->fn(int)->int} fn add(x:int)->int{return (+ x 1)} shadow '
                 'add{assert (== (add 1) 2)} fn main()->int{let f:fn(int)->int=handle {perform '
                 'E.ask(add)} with {ask f->{f}} return (f 6)}',
                 True),
 'callback_wrong_argument': ('effect E{ask:fn(int)->int->int} fn bad(x:bool)->int{return 7} shadow '
                             'bad{assert (== (bad true) 7)} fn main()->int{return handle {perform '
                             'E.ask(bad)} with {ask f->{7}}}',
                             False),
 'enum_array_wrong_argument': ('enum A{One,Two} enum B{One,Two} effect E{ask:array<A>->int} fn '
                               'main()->int{return handle {perform E.ask([B.One])} with {ask a->{7}}}',
                               False),
 'enum_integer_argument': ('enum A{One,Two} effect E{ask:A->A} fn main()->int{let a:A=handle {perform '
                           'E.ask(1)} with {ask a->{a}} return 0}',
                           False),
 'enum_integer_resume': ('enum A{One,Two} effect E{ask:A->A} fn main()->int{let a:A=handle {perform '
                         'E.ask(A.One)} with {ask a->{1}} return 0}',
                         False),
 'enum_tuple_wrong_argument': ('enum A{One,Two} enum B{One,Two} effect E{ask:(A,int)->int} fn '
                               'main()->int{return handle {perform E.ask((B.One,7))} with {ask a->{7}}}',
                               False),
 'enum_wrong_resume': ('enum A{One,Two} enum B{One,Two} effect E{ask:A->A} fn main()->int{let '
                       'a:A=handle {perform E.ask(A.One)} with {ask a->{B.Two}} return 0}',
                       False),
 'generic_wrong_resume': ('union Box<T>{Some{value:T}} effect E{ask:Box<int>->Box<int>} fn '
                          'main()->int{let a:Box<int> = handle {perform E.ask(Box<int>.Some{value:7})} '
                          'with {ask a->{Box<bool>.Some{value:true}}} return 0}',
                          False)})

class EffectSignatureIdentity(unittest.TestCase):
    compile = effects.SelfhostEffects.compile
    execute_module = effects.SelfhostEffects.execute_module

    def test_c_generic_union_array_identity(self):
        # I retain my C frontend regression independently of missing native union arrays.
        driver = str(ROOT/'bin/nano_virt')
        for argument, accepted in [('Box<int>.Some{value:7}', True),
                                   ('Box<bool>.Some{value:true}', False)]:
            with self.subTest(argument=argument), tempfile.TemporaryDirectory() as directory:
                work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                source.write_text('''union Box<T>{Some{value:T}}
effect E{ask:array<Box<int>>->array<Box<int>>}
fn main()->int {
 let values:array<Box<int>> = handle {perform E.ask(['''+argument+'''])} with {ask values->{values}}
 return (array_length values)
}
shadow main{assert (== (main) 1)}
''')
                output.write_bytes(b'prior artifact')
                result = self.compile(driver, source, output)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    executed = subprocess.run([str(ROOT/'bin/nano_vm'), str(output)],
                                              capture_output=True, text=True, timeout=60)
                    self.assertEqual(executed.returncode, 1, executed.stdout+executed.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_nominal_and_generic_execution(self):
        for name, (body, accepted) in CASES.items():
            if not accepted:
                continue
            expected = 0 if name == 'enum_ok' else 7
            for driver in DRIVERS:
                with self.subTest(case=name, driver=driver), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                    source.write_text(body+'\nshadow main{assert (== (main) '+str(expected)+')}\n')
                    result = self.compile(driver, source, output)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.execute_module(output, expected)

    def test_imported_record_effect_execution(self):
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as directory:
                work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                (work/'effects.nano').write_text('''pub struct Point{x:int}
pub effect Move{move:Point->Point}
pub fn send(value:Point)->Point{return perform Move.move(value)}
shadow send{assert true}
''')
                source.write_text('''module "effects.nano" as lib
fn main()->int{let p:lib.Point=handle {(lib.send lib.Point{x:7})} with {move p->{p}} return p.x}
shadow main{assert (== (main) 7)}
''')
                result = self.compile(driver, source, output)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.execute_module(output, 7)

    def test_mismatches_preserve_output(self):
        for name, (body, accepted) in CASES.items():
            if accepted:
                continue
            for driver in DRIVERS:
                with self.subTest(case=name, driver=driver), tempfile.TemporaryDirectory() as directory:
                    work = Path(directory); source = work/'source.nano'; output = work/'source.nvm'
                    source.write_text(body+'\nshadow main{assert true}\n')
                    output.write_bytes(b'prior artifact')
                    result = self.compile(driver, source, output)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

if __name__ == '__main__':
    unittest.main()
