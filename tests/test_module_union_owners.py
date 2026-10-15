"""I retain union declaration owners across module aliases and generic values."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_cseed_generic_functions as base

A = '''pub union Box<T> {Some {value:T}, None {}}
pub fn make()->Box<int>{return Box<int>.Some {value:7}}
shadow make {assert true}
pub fn read(value:Box<int>)->int{match value{Some(p)=>{return p.value} None(p)=>{return 0}}}
shadow read {assert (== (read (make)) 7)}
'''
B = '''pub union Box<T> {Some {value:T, extra:int}, None {}}
pub fn make()->Box<int>{return Box<int>.Some {value:9,extra:100}}
shadow make {assert true}
pub fn read(value:Box<int>)->int{match value{Some(p)=>{return (+ p.value p.extra)} None(p)=>{return 0}}}
shadow read {assert (== (read (make)) 109)}
'''
IMPORTS = 'module "a.nano" as a\nmodule "b.nano" as b\n'
IDENTITY = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
SHADOW = '\nshadow main {assert (== (main) 0)}\n'


class ModuleUnionOwners(unittest.TestCase):
    run_command = base.CseedGenericFunctions.run_command
    execute = base.CseedGenericFunctions.execute

    def test_distinct_layouts_and_both_import_orders(self):
        for imports in (IMPORTS, 'module "b.nano" as b\nmodule "a.nano" as a\n'):
            with self.subTest(imports=imports):
                self.execute(imports+IDENTITY+'''fn main()->int {
    let left:a.Box<int> = (identity (a.make))
    let right:b.Box<int> = (identity (b.make))
    assert (== (a.read left) 7)
    assert (== (b.read right) 109)
    return 0
}'''+SHADOW, files={'a.nano':A,'b.nano':B})

    def test_same_owner_aliases_and_imported_callbacks(self):
        self.execute(IMPORTS+'module "a.nano" as again\nfrom "a.nano" import make as first\n'+IDENTITY+'''fn main()->int {
    let value:again.Box<int> = (identity (first))
    let reader:fn(a.Box<int>)->int = again.read
    let factory:fn()->again.Box<int> = a.make
    assert (== (reader value) 7)
    assert (== (a.read (factory)) 7)
    assert (== (b.read (b.make)) 109)
    return 0
}'''+SHADOW, files={'a.nano':A,'b.nano':B})

    def test_root_union_and_qualified_constructors(self):
        self.execute(IMPORTS+IDENTITY+'''union Box<T> {Some {value:T}, None {}}
fn read(value:Box<int>)->int{match value{Some(p)=>{return p.value} None(p)=>{return 0}}}
shadow read {assert (== (read Box<int>.Some {value:3}) 3)}
fn main()->int {
    let root:Box<int> = Box.Some {value:3}
    let left:a.Box<int> = a.Box<int>.Some {value:7}
    let right:b.Box<int> = b.Box<int>.Some {value:9,extra:100}
    assert (== (read (identity root)) 3)
    assert (== (a.read (identity left)) 7)
    assert (== (b.read (identity right)) 109)
    return 0
}'''+SHADOW, files={'a.nano':A,'b.nano':B})

    def test_ordinary_module_unions(self):
        a=A.replace('Box<T>', 'Box').replace('Box<int>', 'Box').replace('value:T', 'value:int')
        b=B.replace('Box<T>', 'Box').replace('Box<int>', 'Box').replace('value:T', 'value:int')
        self.execute(IMPORTS+IDENTITY+'''fn main()->int {
    let left:a.Box = a.Box.Some {value:7}
    let right:b.Box = b.Box.Some {value:9,extra:100}
    assert (== (a.read (identity left)) 7)
    assert (== (b.read (identity right)) 109)
    return 0
}'''+SHADOW, files={'a.nano':a,'b.nano':b})

    def test_cross_owner_refusals_preserve_output(self):
        bodies = [
            'fn unused()->int{let wrong:Box<int> = (a.make) return 0}',
            'fn unused()->int{let wrong:missing.Box<int> = (a.make) return 0}',
            'fn unused()->int{let wrong:a.Box<int> = (b.make) return 0}',
            'fn unused()->a.Box<int>{return (b.make)}',
            'fn unused()->int{return (a.read (b.make))}',
            'fn unused()->int{let wrong:fn()->a.Box<int> = b.make return 0}',
            'fn choose(x:T,y:T)->T{return x}\nshadow choose {assert (== (choose 1 2) 1)}\n'
            'fn unused()->int{let wrong = (choose (a.make) (b.make)) return 0}',
        ]
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                work=Path(tmp)
                (work/'a.nano').write_text(A)
                # I require distinct owners even when their declarations have identical layouts.
                (work/'b.nano').write_text(A)
                source,output=work/'main.nano',work/'output'
                source.write_text(IMPORTS+body+'\nshadow unused {assert true}\nfn main()->int{return 0}'+SHADOW)
                output.write_bytes(b'prior artifact')
                result=subprocess.run([base.COMPILER,source,*base.MODE,'-o',output],cwd=base.ROOT,
                                      text=True,capture_output=True,timeout=120)
                self.assertEqual(result.returncode,1,result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior artifact')
                self.assertNotIn('already defined',result.stdout+result.stderr)
                self.assertNotIn('C compilation failed',result.stdout+result.stderr)

    def test_malformed_qualified_constructors_preserve_output(self):
        for expression in ('a.Box<int>.Some {value:', 'a.Box<int>. {value:7}',
                           'a.Box<int.Some {value:7}', 'a.Box.Some {value:}'):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                (work/'a.nano').write_text(A)
                source, output = work/'main.nano', work/'output'
                source.write_text('module "a.nano" as a\nfn main()->int{let value = '+expression+' return 0}')
                output.write_bytes(b'prior artifact')
                result = subprocess.run([base.COMPILER, source, *base.MODE, '-o', output],
                                        cwd=base.ROOT, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_duplicate_declaration_in_one_owner_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work/'a.nano').write_text(A+'\npub union Box<T> { Other {value:T} }\n')
            source, output = work/'main.nano', work/'output'
            source.write_text('module "a.nano" as a\nfn main()->int{return 0}'+SHADOW)
            output.write_bytes(b'prior artifact')
            result = subprocess.run([base.COMPILER, source, *base.MODE, '-o', output],
                                    cwd=base.ROOT, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(), b'prior artifact')
