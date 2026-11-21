"""I keep imported constructors distinct from a caller's same-spelled record."""
from pathlib import Path
import os,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked,source_compiler_command
from tests.native_process import run_captured
ROOT=Path(__file__).resolve().parents[1]
LIBRARY='''module witness_library
pub struct Function { value: int }
pub fn make() -> Function { return Function { value: 42 } }
shadow make { assert (== (make).value 42) }
'''
WITNESS='''module witness_caller
module "lib.nano" as library
pub struct Function { other: int, marker: int }
fn forward() -> library.Function { return library.Function { value: 42 } }
shadow forward { assert (== (forward).value 42) }
fn main() -> int { (println (forward).value) return 0 }
shadow main { assert (== (main) 0) }
'''
CONTAINERS='''module witness_containers
module "lib.nano" as library
pub struct Function { other: string }
struct Envelope { values: array<library.Function> }
fn forward() -> array<library.Function> { return [library.Function { value: 42 }, (library.make)] }
shadow forward { assert (== (at (forward) 0).value 42) }
fn main() -> int {
 let value: Envelope = Envelope { values: (forward) }
 assert (== (at value.values 1).value 42)
 (println (at value.values 0).value)
 return 0
}
shadow main { assert (== (main) 0) }
'''

BRANCHES='''module witness_branches
module "lib.nano" as library
pub struct Function { other: int }
union Choice { Some { n: int }, Empty {} }
fn conditional(flag: bool) -> library.Function {
 return (if flag { library.Function { value: 42 } } else { (library.make) })
}
shadow conditional { assert (== (conditional true).value 42) assert (== (conditional false).value 42) }
fn selection(flag: bool) -> library.Function { return (cond (flag (library.make)) (else library.Function { value: 42 })) }
shadow selection { assert (== (selection false).value 42) assert (== (selection true).value 42) }
fn matching(value: Choice) -> library.Function { return match value { Some(item) => library.Function { value: item.n } Empty(empty) => { (library.make) } } }
shadow matching { assert (== (matching Choice.Some { n: 42 }).value 42) assert (== (matching Choice.Empty {}).value 42) }
fn anonymous() -> library.Function { return { value: 42 } }
shadow anonymous { assert (== (anonymous).value 42) }
fn local() -> library.Function { let value: library.Function = (library.make) return value }
shadow local { assert (== (local).value 42) }
let mut probes: int = 0
fn condition()->bool { set probes (+ probes 1) return false }
shadow condition { set probes 0 assert (not (condition)) assert (== probes 1) }
fn skipped()->library.Function { set probes (+ probes 100) return (library.make) }
shadow skipped { set probes 0 assert (== (skipped).value 42) assert (== probes 100) }
fn once()->library.Function {
 set probes 0
 let selected: library.Function = (if (condition) { (skipped) } else { let values: array<int> = [42] set probes (+ probes 10) library.Function { value: (at values 0) } })
 assert (== probes 11)
 return selected
}
shadow once { assert (== (once).value 42) }
fn lexical(flag:bool)->library.Function { return (if flag { return (library.make) } else { library.Function { value: 42 } }) }
shadow lexical { assert (== (lexical true).value 42) assert (== (lexical false).value 42) }
fn main()->int { assert (== (once).value 42) assert (== (lexical true).value 42) assert (== (matching Choice.Empty {}).value 42) assert (== (conditional false).value 42) assert (== (anonymous).value 42) assert (== (local).value 42) (println (selection true).value) return 0 }
shadow main { assert (== (main) 0) }
'''

class ImportedConstructorIdentity(unittest.TestCase):
    def test_original_witness_and_nested_containers(self):
        for name,source in [('unchanged',WITNESS),('containers',CONTAINERS),('branches',BRANCHES)]:
            with self.subTest(case=name),tempfile.TemporaryDirectory(prefix='constructor-identity-') as directory:
                root=Path(directory);(root/'lib.nano').write_text(LIBRARY);path=root/'main.nano';path.write_text(source)
                for route,command in build_routes(path,root):
                    with self.subTest(route=route):self.assertEqual(checked(command),'42\n')
    def test_same_shape_distinct_owner_refuses_before_publication(self):
        snippets=[
            'fn take(value: library.Function) -> int { return value.value } shadow take { assert (== (take (library.make)) 42) } fn main()->int { return (take Function { value: 42 }) }',
            'fn wrong() -> library.Function { return Function { value: 42 } } shadow wrong { assert true } fn main()->int { return 0 }',
            'fn main()->int { let values: array<library.Function> = [Function { value: 42 }] return 0 }',
            'fn wrong()->library.Function { let value: Function = Function {value:42} return value } shadow wrong {assert true} fn main()->int {return 0}',
            'fn other()->Function {return Function {value:42}} shadow other {assert true} fn wrong()->library.Function {return (other)} shadow wrong {assert true} fn main()->int {return 0}',
            'fn wrong(flag:bool)->library.Function {return (if flag {(library.make)} else {Function {value:42}})} shadow wrong {assert true} fn main()->int {return 0}',
            'fn wrong(flag:bool)->library.Function {return (cond (flag (library.make)) (else Function {value:42}))} shadow wrong {assert true} fn main()->int {return 0}',

        ]
        compilers=[('c',[ROOT/'bin/nanoc_c'],[]),('vm',[ROOT/'bin/nano_virt'],['--emit-nvm'])]
        source = source_compiler_command()
        if source:compilers.append(('nanolang',source,['--emit-nvm']))
        with tempfile.TemporaryDirectory(prefix='constructor-refusals-') as directory:
            root=Path(directory);(root/'lib.nano').write_text(LIBRARY);path=root/'main.nano';output=root/'prior'
            for i,snippet in enumerate(snippets):
                path.write_text('module witness_caller\nmodule "lib.nano" as library\nstruct Function { value: int }\n'+snippet+'\nshadow main { assert true }\n')
                for name,compiler,flags in compilers:
                    with self.subTest(case=i,producer=name):
                        output.write_bytes(b'prior artifact\0');result=run_captured([*compiler,path,*flags,'-o',output],build=True,timeout=120,cwd=ROOT)
                        self.assertNotEqual(result.returncode,0);self.assertEqual(output.read_bytes(),b'prior artifact\0')
                        self.assertNotIn('C compilation failed',(result.stdout+result.stderr).decode(errors='replace'))

if __name__=='__main__':unittest.main()
