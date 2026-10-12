"""I retain imported effect ownership and refuse invisible operations."""
import os
from pathlib import Path
import tempfile
import unittest
from tests import test_selfhost_effects as effects

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = os.environ.get('NANOLANG_IMPORTED_EFFECT_COMPILERS', str(ROOT/'bin/nano_virt')).split(os.pathsep)
PUBLIC = 'pub effect E{ask:int->int}\n'
PRIVATE = 'effect Secret{ask:int->int}\npub fn marker()->int{return 0}\nshadow marker{assert (== (marker) 0)}\n'


class ImportedEffectIdentity(unittest.TestCase):
    compile = effects.SelfhostEffects.compile
    execute_module = effects.SelfhostEffects.execute_module

    def check(self, files, body, accepted=True, expected=7):
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                for name, content in files.items():
                    (work/name).write_text(content)
                source = work/'main.nano'; output = work/'main.nvm'
                source.write_text(body+'\nshadow main{assert (== (main) '+str(expected)+')}\n')
                output.write_bytes(b'prior artifact')
                result = self.compile(driver, source, output)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.execute_module(output, expected)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_same_named_effects(self):
        files = {
            'left.nano': 'pub effect E{left_op:int->int}\npub fn send()->int{return perform E.left_op(3)}\nshadow send{assert true}\n',
            'right.nano': 'pub effect E{right_op:int->int}\npub fn send()->int{return perform E.right_op(5)}\nshadow send{assert true}\n',
        }
        self.check(files, '''module "left.nano" as left
module "right.nano" as right
fn main()->int{let a:int=handle {(left.send)} with {left_op n->{n}} let b:int=handle {(right.send)} with {right_op n->{n}} return (+ a b)}''', expected=8)

    def test_distinct_record_signatures(self):
        files = {
            'left.nano': 'pub struct Point{x:int} pub effect E{left_op:Point->Point} pub fn send()->Point{return perform E.left_op(Point{x:3})} shadow send{assert true}',
            'right.nano': 'pub struct Point{x:int} pub effect E{right_op:Point->Point} pub fn send()->Point{return perform E.right_op(Point{x:5})} shadow send{assert true}',
        }
        self.check(files, '\n'.join([
            'module "left.nano" as left',
            'module "right.nano" as right',
            'fn main()->int{let a:left.Point=handle {(left.send)} with {left_op p->{p}} let b:right.Point=handle {(right.send)} with {right_op p->{p}} return (+ a.x b.x)}',
        ]), expected=8)

    def test_distinct_argument_and_result_records(self):
        self.check({'lib.nano': 'pub struct Input{x:int} pub struct Output{text:string,x:int} pub effect E{convert:Input->Output}'}, '\n'.join([
            'module "lib.nano" as lib',
            'fn main()->int{let out:lib.Output=handle {perform lib.E.convert(lib.Input{x:7})} with {convert p->{lib.Output{text:"kept",x:p.x}}} assert (== out.text "kept") return out.x}',
        ]))

    def test_qualified_perform(self):
        self.check({'lib.nano': PUBLIC}, '''module "lib.nano" as lib
fn main()->int{return handle {perform lib.E.ask(7)} with {ask n->{n}}}''')

    def test_selective_alias(self):
        self.check({'lib.nano': PUBLIC}, '''from "lib.nano" import E as Chosen
fn main()->int{return handle {perform Chosen.ask(7)} with {ask n->{n}}}''')

    def test_private_effect_remains_local(self):
        self.check({'lib.nano': '''effect Secret{ask:int->int}
pub fn send()->int{return handle {perform Secret.ask(7)} with {ask n->{n}}}
shadow send{assert (== (send) 7)}
'''}, '''module "lib.nano" as lib
fn main()->int{return (lib.send)}''')

    def test_private_effect_does_not_ambiguate_local_handler(self):
        self.check({'lib.nano': PRIVATE}, '''module "lib.nano" as lib
effect Local{ask:int->int}
fn main()->int{return handle {perform Local.ask(7)} with {ask n->{n}}}''')

    def test_private_imports_preserve_output(self):
        for header, target in [
            ('module "lib.nano" as lib', 'Secret'),
            ('module "lib.nano" as lib', 'lib.Secret'),
            ('from "lib.nano" import Secret', 'Secret'),
        ]:
            with self.subTest(target=target, header=header):
                self.check({'lib.nano': PRIVATE}, header+'\nfn main()->int{return handle {perform '+target+'.ask(7)} with {ask n->{n}}}', accepted=False)

    def test_private_handler_preserves_output(self):
        self.check({'lib.nano': '''effect Secret{ask:int->int}
pub fn send()->int{return perform Secret.ask(7)}
shadow send{assert true}
'''}, '''module "lib.nano" as lib
fn main()->int{return handle {(lib.send)} with {ask n->{n}}}''', accepted=False)

    def test_distinct_record_mismatches_preserve_output(self):
        files = {
            'left.nano': 'pub struct Point{x:int} pub effect E{left_op:Point->Point}',
            'right.nano': 'pub struct Point{x:int} pub effect E{right_op:Point->Point}',
        }
        for argument, resumed in [('right.Point{x:7}', 'p'), ('left.Point{x:7}', 'right.Point{x:7}')]:
            with self.subTest(argument=argument, resumed=resumed):
                self.check(files, 'module "left.nano" as left\nmodule "right.nano" as right\nfn main()->int{let a:left.Point=handle {perform left.E.left_op('+argument+')} with {left_op p->{'+resumed+'}} return a.x}', accepted=False)
