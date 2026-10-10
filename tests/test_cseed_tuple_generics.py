"""I preserve recursive tuple identities through generic source execution."""
import subprocess
import tempfile
from pathlib import Path
from tests import test_cseed_generic_functions as base

IDENTITY = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
MAIN_SHADOW = '\nshadow main {assert (== (main) 0)}\n'


class TupleGenerics(base.CseedGenericFunctions):
    def test_tuple_identity_and_destructuring(self):
        self.execute(IDENTITY+'''fn main()->int {
    let (number, flag) = (identity (identity (7, true)))
    assert (== number 7)
    assert flag
    return 0
}'''+MAIN_SHADOW)

    def test_nested_tuple_and_array_results(self):
        self.execute(IDENTITY+'''fn make()->((int,bool),array<string>){return ((7,true),["kept"])}
shadow make {let value = (make) assert (value.0).1}
fn main()->int {
    let value:((int,bool),array<string>) = (identity (make))
    assert (== (value.0).0 7)
    assert (value.0).1
    assert (== (at value.1 0) "kept")
    return 0
}'''+MAIN_SHADOW)

    def test_structural_tuple_parameter(self):
        self.execute('''fn first(value:(T,E))->T{return value.0}
shadow first {assert (== (first (7,true)) 7)}
fn main()->int {
    assert (== (first (7,true)) 7)
    assert (== (first ("kept",[9])) "kept")
    return 0
}'''+MAIN_SHADOW)

    def test_tuple_nominal_and_callback_elements(self):
        self.execute(IDENTITY+'''enum Shade {Dark=2}
struct Box {value:int}
fn increment(value:int)->int{return (+ value 1)}
shadow increment {assert (== (increment 4) 5)}
fn main()->int {
    let value:(Box,Shade,fn(int)->int) = (identity (Box {value:7},Shade.Dark,increment))
    assert (== value.0.value 7)
    assert (== (identity value.1) 2)
    let callback:fn(int)->int = value.2
    assert (== (callback 4) 5)
    return 0
}'''+MAIN_SHADOW)

    def test_tuple_binding_mismatches_preserve_output(self):
        for left,right in [('(1,true)','(1,2)'),('((1,true),3)','((1,2),3)'),
                           ('([1],true)','([false],true)'),('(1,true)','(1,true,3)')]:
            with self.subTest(left=left,right=right), tempfile.TemporaryDirectory() as tmp:
                source,output = Path(tmp)/'main.nano',Path(tmp)/'main.nvm'
                output.write_bytes(b'prior')
                source.write_text('fn choose(a:T,b:T)->T{return a}\nshadow choose {assert (== (choose 1 2) 1)}\n'
                    +f'fn main()->int{{let result = (choose {left} {right}) return 0}}'+MAIN_SHADOW)
                result=subprocess.run([str(base.COMPILER),str(source),'--emit-nvm','-o',str(output)],
                                      cwd=base.ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('one concrete identity',result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior')

    def test_declared_tuple_shapes_preserve_output(self):
        programs = [
            'fn bad()->(int,bool){return (1,2)}\nshadow bad {assert true}\nfn main()->int{return 0}',
            'fn main()->int{let value:(int,bool)=(1,2) return 0}',
            'fn main()->int{let mut value:(int,bool)=(1,true) set value (1,2) return 0}',
            'fn take(value:(int,bool))->int{return 0}\nshadow take {assert true}\nfn main()->int{return (take (1,2))}',
        ]
        for program in programs:
            with self.subTest(program=program), tempfile.TemporaryDirectory() as tmp:
                source,output=Path(tmp)/'main.nano',Path(tmp)/'main.nvm'
                output.write_bytes(b'prior')
                source.write_text(program+'\nshadow main {assert true}\n')
                result=subprocess.run([str(base.COMPILER),str(source),'--emit-nvm','-o',str(output)],
                                      cwd=base.ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertRegex(result.stdout+result.stderr, r'one complete tuple shape|expected a value of type `\(int,bool\)`, but found `\(int,int\)`|expected \(int,bool\), got \(int,int\)')
                self.assertEqual(output.read_bytes(),b'prior')

    def test_structural_tuple_result_substitution(self):
        self.execute('''fn swap(value:(T,E))->(E,T){return (value.1,value.0)}
shadow swap {let value=(swap (7,true)) assert value.0 assert (== value.1 7)}
fn main()->int {
    let value:(int,string) = (swap ("kept",7))
    assert (== value.0 7)
    assert (== value.1 "kept")
    let nested:(bool,array<int>) = (swap ([9],true))
    assert nested.0
    assert (== (at nested.1 0) 9)
    return 0
}'''+MAIN_SHADOW)

    def test_imported_tuple_result(self):
        self.execute('module "pair.nano" as pair\n'+IDENTITY+'''fn main()->int {
    let value = (identity (pair.make))
    assert value.1
    assert (== value.0.value 7)
    return 0
}'''+MAIN_SHADOW, files={'pair.nano': '''pub struct Box {value:int}
pub fn make()->(Box,bool){return (Box {value:7},true)}
shadow make {let value = (make) assert value.1}
'''})

    def test_tuple_projection_bounds_preserve_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,output=Path(tmp)/'main.nano',Path(tmp)/'main.nvm'
            output.write_bytes(b'prior')
            source.write_text('fn pair()->(int,bool){return (1,true)}\nshadow pair {assert true}\n'
                'fn main()->int{let value=(pair).2 return 0}\nshadow main {assert true}\n')
            result=subprocess.run([str(base.COMPILER),str(source),'--emit-nvm','-o',str(output)],
                                  cwd=base.ROOT,capture_output=True,text=True,timeout=120)
            self.assertGreater(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(),b'prior')
