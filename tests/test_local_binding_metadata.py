"""I preserve optional lexical names from both ordinary source frontends."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(path) for path in os.environ.get('NANOLANG_LOCAL_NAME_COMPILERS',
    os.pathsep.join(str(ROOT/'bin'/name) for name in ('nano_virt','nanoc_stage1','nanoc_stage2'))).split(os.pathsep) if path]
if not DRIVERS:
    raise ValueError('I require at least one local-name producer')
SOURCE = '''fn compute(flag: bool) -> int {
    let value: int = 7
    if flag { let value: int = 11 assert (== value 11) }
    else { let value: int = 12 assert (== value 12) }
    let mut count: int = 0
    while (< count 2) { let step: int = (+ count 1) set count step }
    return (+ value count)
}
shadow compute { assert (== (compute true) 9) assert (== (compute false) 9) }
fn main() -> int { return (compute true) }
shadow main { assert (== (main) 9) }
'''
ARRAY = '''fn main() -> int {
    let values: array<int> = [1, 2]
    let answer: int = (array_length values)
    return answer
}
shadow main { assert (== (main) 2) }
'''

class LocalBindingMetadata(unittest.TestCase):
    def run_checked(self, args, expected=0):
        env = os.environ.copy()
        env.setdefault("NANOLANG_SHADOW_TIMEOUT_SECONDS", "60")
        p = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout

    def produce(self, source, expected, inspect, files=None, debug_line=None):
        with tempfile.TemporaryDirectory(prefix="nano-local-names-") as name:
            tmp=Path(name); input_file=tmp/"case.nano"; input_file.write_text(source)
            for filename, contents in (files or {}).items():
                (tmp/filename).write_text(contents)
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    artifact=tmp/f"{driver.name}.nvm"
                    self.run_checked([driver,input_file,"--emit-nvm","-o",artifact])
                    # The C probe validates every interval, exact metadata/code
                    # preservation, original wire equality and canonical-cycle stability.
                    lines=self.run_checked([ROOT/"obj/test_local_bindings",artifact]+(["--require-debug"] if driver.name=="nano_virt" else [])).splitlines()
                    bindings=[(f,int(slot),int(begin),int(end),local) for f,slot,begin,end,local in (line.split() for line in lines)]
                    inspect(bindings)
                    self.run_checked([ROOT/"bin/nano_vm",artifact],expected)
                    native=tmp/f"{driver.name}.c"; binary=tmp/f"{driver.name}.out"
                    self.run_checked([ROOT/"bin/nvm2c",artifact,"-o",native])
                    self.run_checked(shlex.split(os.environ.get("CC","cc"))+["-std=c11","-Wall","-Wextra","-Werror",str(native),"-lm","-o",str(binary)])
                    self.run_checked([binary],expected)
                    # I remove only advisory names, preserving other metadata and all strings.
                    canonical=self.run_checked([ROOT/"bin/nanoisa","dump",artifact]).splitlines()
                    if debug_line is not None and driver.name=='nano_virt':
                        locations=[line.split() for line in canonical if line.startswith('.debug ')]
                        self.assertEqual(locations[0][1:3],['0',str(debug_line)])
                    strings=[line for line in canonical if line.startswith('.string ')]
                    local_key=str(strings.index('.string "nano.local.v1"'))
                    stripped=[line for line in canonical if not
                              (line.startswith('.metadata ') and line.split()[1]==local_key)]
                    self.assertLess(len(stripped),len(canonical))
                    assembly=tmp/'stripped.nasm'; unnamed=tmp/'stripped.nvm'; unnamed_c=tmp/'stripped.c'
                    assembly.write_text('\n'.join(stripped)+'\n')
                    self.run_checked([ROOT/"bin/nanoisa","asm",assembly,"-o",unnamed])
                    self.run_checked([ROOT/"bin/nano_vm",unnamed],expected)
                    self.run_checked([ROOT/"bin/nvm2c",unnamed,"-o",unnamed_c])
                    self.assertEqual(native.read_bytes(),unnamed_c.read_bytes())

    def test_parameters_shadowed_bindings_and_loop(self):
        def inspect(bindings):
            records=[b for b in bindings if b[0]=="compute"]
            self.assertEqual(sorted(b[4] for b in records),["count","flag","step","value","value","value"])
            flag=next(b for b in records if b[4]=="flag")
            self.assertEqual(flag[1:3],(0,0))
            values=sorted((b for b in records if b[4]=="value"),key=lambda b:b[2])
            self.assertEqual(len({b[1] for b in values}),3)
            self.assertLess(values[0][2],values[1][2])
            self.assertGreaterEqual(values[0][3],values[2][3])
            self.assertLessEqual(values[1][3],values[2][2])
            step=next(b for b in records if b[4]=="step")
            self.assertLess(step[2],step[3])
        self.produce(SOURCE,9,inspect)

    def test_array_name_is_retained(self):
        def inspect(bindings):
            names=[b[4] for b in bindings if b[0]=="main"]
            self.assertEqual(sorted(names),["answer","values"])
        self.produce(ARRAY,2,inspect)

    def test_aggregate_and_callable_parameters_and_locals(self):
        source = '''struct Item {value:int}
enum Color {Red,Blue}
union Choice {Some {value:int}}
fn identity(value:int)->int{return value}
shadow identity {assert (== (identity 7) 7)}
fn inspect(values:array<int>,item:Item,pair:(int,bool),color:Color,choice:Choice,callback:fn(int)->int)->int {
 let copy:Item = item
 let tuple:(int,bool) = pair
 let selected:Color = color
 let variant:Choice = choice
 let function:fn(int)->int = callback
 assert (== (function (at values 0)) 7)
 assert (== copy.value 9)
 assert tuple.1
 assert (== selected Color.Blue)
 return 0
}
shadow inspect {assert (== (inspect [7] Item {value:9} (7,true) Color.Blue Choice.Some {value:7} identity) 0)}
fn main()->int {return (inspect [7] Item {value:9} (7,true) Color.Blue Choice.Some {value:7} identity)}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            entries=[b for b in bindings if b[0]=='inspect']
            self.assertEqual(sorted(b[4] for b in entries), sorted(['values','item','pair','color','choice','callback','copy','tuple','selected','variant','function']))
            for name in ['values','item','pair','color','choice','callback']:
                entry=next(b for b in entries if b[4]==name)
                self.assertEqual(entry[2],0)
                self.assertGreater(entry[3],entry[2])
        self.produce(source,0,inspect)

    def test_shadowed_array_intervals(self):
        source='''fn main()->int {
 let values:array<int> = [7]
 if true {let values:array<int> = [9] assert (== (at values 0) 9)}
 assert (== (at values 0) 7)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            entries=sorted((b for b in bindings if b[0]=='main' and b[4]=='values'),key=lambda b:b[2])
            self.assertEqual(len(entries),2)
            self.assertNotEqual(entries[0][1],entries[1][1])
            self.assertLess(entries[0][2],entries[1][2])
            self.assertGreaterEqual(entries[0][3],entries[1][3])
        self.produce(source,0,inspect)

    def test_generated_tuple_owner_is_omitted_without_reserving_names(self):
        source='''fn main()->int {
 let __td99:int = 9
 let (first,second) = (7,"kept")
 assert (== first 7)
 assert (== second "kept")
 assert (== __td99 9)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['__td99','first','second'])
        self.produce(source,0,inspect)

    def test_specialized_array_parameter_keeps_source_name(self):
        source='''fn same(value:T)->T{return value}
shadow same {assert (== (same 1) 1)}
fn main()->int {
 let numbers:array<int> = (same [7])
 let words:array<string> = (same ["kept"])
 assert (== (at numbers 0) 7)
 assert (== (at words 0) "kept")
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            parameters=[b for b in bindings if b[4]=='value' and b[2]==0]
            self.assertGreaterEqual(len({b[0] for b in parameters}),2)
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['numbers','words'])
        self.produce(source,0,inspect)

    def test_imported_generated_owner_and_source_names(self):
        source='''module "values.nano" as values
fn main()->int {return (values.take [7])}
shadow main {assert (== (main) 16)}
'''
        module='''pub fn take(values:array<int>)->int {
 let (first,second) = ((at values 0),9)
 return (+ first second)
}
shadow take {assert (== (take [7]) 16)}
'''
        def inspect(bindings):
            owners={b[0] for b in bindings if b[4]=='values'}
            self.assertEqual(len(owners),1)
            self.assertEqual(sorted(b[4] for b in bindings if b[0] in owners),['first','second','values'])
        self.produce(source,16,inspect,files={'values.nano':module},debug_line=2)

    def test_global_initializer_lambda_and_imported_passive_offsets(self):
        source = '''import "values.nano" as values
let base:int = 7
fn main()->int {
 let transform:fn(int)->int = fn(value:int)->int {return (+ value 1)}
 let result:int = (values.compute (transform base))
 assert (== result 25)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        module = '''pub fn compute(input:int)->int {
 par {let left:int = (+ input 1) let right:int = (* input 2)}
 return (+ left right)
}
shadow compute {assert (== (compute 8) 25)}
'''
        def inspect(bindings):
            names=[b[4] for b in bindings if b[0]=='main']
            self.assertEqual(sorted(names),['result','transform'])
        self.produce(source,0,inspect,files={'values.nano':module})

if __name__=="__main__": unittest.main()
