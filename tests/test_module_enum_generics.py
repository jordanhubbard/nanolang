"""I retain module-owned enums through generic checking and execution."""
import os,shlex,sys,unittest,subprocess,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANO_ENUM_COMPILER", ROOT / "bin/nanoc_stage2"))
class EnumGenerics(unittest.TestCase):
    def run_command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-generic-') as tmp:
            work = Path(tmp)
            for name, contents in (files or {}).items():
                (work / name).write_text(contents)
            path, module, c, binary = [work / name for name in ('main.nano', 'main.nvm', 'main.c', 'main')]
            path.write_text(source)
            self.run_command([COMPILER, path, '--emit-nvm', '-o', module])
            self.run_command([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.run_command([ROOT / 'bin/nano_vm', module])
            self.run_command([ROOT / 'bin/nvm2c', module, '-o', c])
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
            self.run_command(cc + ['-std=c11', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                c, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary])
            self.run_command([binary])
    def test_enum_literal_return_local_and_field(self):
        self.execute('''enum Color {RED=0,BLUE=1}
struct Box {color:Color}
fn identity(value:T)->T{let copy:T=value return copy}
shadow identity {assert (== (identity 3) 3)}
fn make()->Color{return Color.BLUE}
shadow make {assert (== (make) Color.BLUE)}
fn main()->int{
 let color:Color = (identity (identity Color.BLUE))
 assert (== color Color.BLUE)
 assert (== (identity color) Color.BLUE)
 assert (== (identity (make)) Color.BLUE)
 let box:Box = Box {color:Color.RED}
 assert (== (identity box.color) Color.RED)
 return 0
}
shadow main {assert (== (main) 0)}
''')
    def test_enum_array_identity(self):
        self.execute('''enum Color {RED=0,BLUE=1}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 3) 3)}
fn choose(a:T,b:T)->T{return a}
shadow choose {assert (== (choose 1 2) 1)}
fn main()->int{
 let colors:array<Color> = [Color.BLUE]
 let copied:array<Color> = (choose colors [Color.RED])
 assert (== (at (identity copied) 0) Color.BLUE)
 return 0
}
shadow main {assert (== (main) 0)}
''')
    def test_distinct_enum_bindings_refuse_during_checking(self):
        source=(ROOT/'docs/evidence/generic-enum-tuple-baseline-20261010/enum-mismatch.nano').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path,output=Path(tmp)/'main.nano',Path(tmp)/'out.nvm';path.write_text(source);output.write_bytes(b'prior')
            result=subprocess.run([str(COMPILER),str(path),'--emit-nvm','-o',str(output)],capture_output=True,text=True,cwd=ROOT,timeout=120)
            self.assertGreater(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('one concrete identity',result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(),b'prior')
    def test_imported_enum_results(self):
        self.execute('module "a.nano" as a\nmodule "b.nano" as b\n'
            'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
            'fn main()->int{assert (== (identity (a.make)) 1) assert (== (identity (b.make)) 2) return 0}\n'
            'shadow main {assert (== (main) 0)}\n', files={
            'a.nano':'pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n',
            'b.nano':'pub enum Shade {VALUE=2}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 2)}\n'})

    def test_same_named_imported_enums_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            for name in ('a','b'):
                (work/(name+'.nano')).write_text('pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n')
            source=work/'main.nano';output=work/'main.nvm';output.write_bytes(b'prior')
            source.write_text('module "a.nano" as a\nmodule "b.nano" as b\n'
                'fn choose(a:T,b:T)->T{return a}\nshadow choose {assert (== (choose 1 2) 1)}\n'
                'fn main()->int{let chosen = (choose (a.make) (b.make)) return 0}\nshadow main {assert (== (main) 0)}\n')
            r=subprocess.run([str(COMPILER),str(source),'--emit-nvm','-o',str(output)],capture_output=True,text=True,cwd=ROOT,timeout=120)
            self.assertGreater(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('one concrete identity',r.stdout+r.stderr)
            self.assertEqual(output.read_bytes(),b'prior')

    def test_qualified_result_annotations_and_selective_alias(self):
        self.execute('module "a.nano" as a\nfrom "b.nano" import Shade as Other, make as other_make\n'
            'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
            'fn main()->int{let first:a.Shade = (identity (a.make)) let second:Other = (identity (other_make)) assert (== first 1) assert (== second 2) return 0}\n'
            'shadow main {assert (== (main) 0)}\n', files={
            'a.nano':'pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n',
            'b.nano':'pub enum Shade {VALUE=2}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 2)}\n'})

    def test_local_record_shadows_enum_receiver(self):
        self.execute("""enum Shade {VALUE=1}
struct Box {VALUE:int}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 1) 1)}
fn main()->int{let Shade:Box = Box {VALUE:8} assert (== (identity Shade.VALUE) 8) return 0}
shadow main {assert (== (main) 0)}
""")

    def test_same_alias_in_separate_importers(self):
        self.execute('module "c.nano" as c\nmodule "d.nano" as d\n'
            'fn main()->int{assert (== (c.get) 1) assert (== (d.get) 2) return 0}\n'
            'shadow main {assert (== (main) 0)}\n', files={
            'a.nano':'pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n',
            'b.nano':'pub enum Shade {VALUE=2}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 2)}\n',
            'c.nano':'module "a.nano" as lib\nfrom "a.nano" import Shade as Chosen\npub fn get()->Chosen{return (lib.make)}\nshadow get {assert (== (get) 1)}\n',
            'd.nano':'module "b.nano" as lib\nfrom "b.nano" import Shade as Chosen\npub fn get()->Chosen{return (lib.make)}\nshadow get {assert (== (get) 2)}\n'})

    def test_root_and_imported_enums_remain_separate(self):
        self.execute('module "a.nano" as a\nenum Shade {VALUE=9}\nenum A {VALUE=7}\n'
            'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'
            'fn main()->int{assert (== (identity Shade.VALUE) 9) assert (== (identity (a.make)) 1) assert (== (identity A.VALUE) 7) return 0}\n'
            'shadow main {assert (== (main) 0)}\n', files={
            'a.nano':'pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n'})

    def test_root_and_imported_enum_bindings_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work/'a.nano').write_text('pub enum Shade {VALUE=1}\npub fn make()->Shade{return Shade.VALUE}\nshadow make {assert (== (make) 1)}\n')
            source, output = work/'main.nano', work/'main.nvm'
            output.write_bytes(b'prior')
            source.write_text('module "a.nano" as a\nenum Shade {VALUE=1}\n'
                'fn choose(a:T,b:T)->T{return a}\nshadow choose {assert (== (choose 1 2) 1)}\n'
                'fn main()->int{let value = (choose (a.make) Shade.VALUE) return 0}\nshadow main {assert (== (main) 0)}\n')
            result = subprocess.run([str(COMPILER),str(source),'--emit-nvm','-o',str(output)],capture_output=True,text=True,cwd=ROOT,timeout=120)
            self.assertGreater(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('one concrete identity',result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(),b'prior')

if __name__=='__main__': unittest.main()
