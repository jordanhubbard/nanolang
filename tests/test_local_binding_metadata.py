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

    def produce(self, source, expected, inspect, files=None, debug_line=None, inspect_captures=None):
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
                    if inspect_captures:
                        capture_lines=self.run_checked([ROOT/"obj/test_local_bindings",artifact,"--upvalues"]).splitlines()
                        inspect_captures([(f,int(slot),int(begin),int(end),local) for f,slot,begin,end,local in (line.split() for line in capture_lines)])
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
                    name_keys={str(strings.index(line)) for line in strings if line in ('.string "nano.local.v1"','.string "nano.upvalue.v1"')}
                    stripped=[line for line in canonical if not
                              (line.startswith('.metadata ') and line.split()[1] in name_keys)]
                    self.assertLess(len(stripped),len(canonical))
                    assembly=tmp/'stripped.nasm'; unnamed=tmp/'stripped.nvm'; unnamed_c=tmp/'stripped.c'
                    assembly.write_text('\n'.join(stripped)+'\n')
                    self.run_checked([ROOT/"bin/nanoisa","asm",assembly,"-o",unnamed])
                    self.run_checked([ROOT/"bin/nano_vm",unnamed],expected)
                    self.run_checked([ROOT/"bin/nvm2c",unnamed,"-o",unnamed_c])
                    self.assertEqual(native.read_bytes(),unnamed_c.read_bytes())
                    if '.string "nano.upvalue.v1"' in strings:
                        capture_key=str(strings.index('.string "nano.upvalue.v1"'))
                        duplicate=next(line for line in canonical if line.startswith('.metadata ') and line.split()[1]==capture_key)
                        alternatives=[canonical+[duplicate],
                                      [line.replace('nano.upvalue.v1','nano.upvalue.v2') if line=='.string "nano.upvalue.v1"' else line for line in canonical]]
                        for advisory in alternatives:
                            assembly.write_text('\n'.join(advisory)+'\n')
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

    def test_nested_functions_keep_separate_local_names(self):
        source='''struct Item {value:int}
fn main()->int {
 let base:int = 7
 let outer:fn(int)->fn(int)->int = fn(offset:int)->fn(int)->int {
  let item:Item = Item {value:offset}
  let values:array<int> = [item.value]
  return fn(delta:int)->int {
   let result:int = (+ (+ base (at values 0)) delta)
   if true {let result:int = 99 assert (== result 99)}
   return result
  }
 }
 let inner:fn(int)->int = (outer 3)
 assert (== (inner 2) 12)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['base','inner','outer'])
            outer={b[0] for b in bindings if b[4]=='offset'}
            inner={b[0] for b in bindings if b[4]=='delta'}
            self.assertEqual(len(outer),1)
            self.assertEqual(len(inner),1)
            self.assertTrue(outer.isdisjoint(inner))
            self.assertEqual(sorted(b[4] for b in bindings if b[0] in outer),['item','offset','values'])
            self.assertEqual(sorted(b[4] for b in bindings if b[0] in inner),['delta','result','result'])
            for name in ['offset','delta']:
                self.assertEqual(next(b[2] for b in bindings if b[4]==name),0)
            results=sorted((b for b in bindings if b[0] in inner and b[4]=='result'),key=lambda b:b[2])
            self.assertNotEqual(results[0][1],results[1][1])
            self.assertLess(results[0][2],results[1][2])
            self.assertGreaterEqual(results[0][3],results[1][3])
        def inspect_captures(bindings):
            self.assertEqual(sorted(b[4] for b in bindings),['base','base','values'])
            self.assertEqual(len({b[0] for b in bindings}),2)
            for entry in bindings:
                self.assertEqual(entry[2],0)
                self.assertGreater(entry[3],0)
        self.produce(source,0,inspect,inspect_captures=inspect_captures)

    def test_aggregate_and_callable_capture_names(self):
        source='''struct Item {value:int}
fn main()->int {
 let item:Item = Item {value:7}
 let pair:(int,bool) = (3,true)
 let text:string = "kept"
 let callback:fn(int)->int = fn(input:int)->int {return (+ input 1)}
 let reader:fn()->int = fn()->int {
  assert (== text "kept")
  assert pair.1
  return (callback (+ item.value pair.0))
 }
 assert (== (reader) 11)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['callback','item','pair','reader','text'])
        def captures(bindings):
            self.assertEqual(sorted(b[4] for b in bindings),['callback','item','pair','text'])
            self.assertEqual(len({b[0] for b in bindings}),1)
            self.assertEqual(sorted(b[1] for b in bindings),[0,1,2,3])
        self.produce(source,0,inspect,inspect_captures=captures)

    def test_iteration_names_have_separate_lexical_intervals(self):
        source='''struct Item {value:int}
fn main()->int {
 let item:int = 99
 let mut sum:int = 0
 for item in [1,2,3] {
  if (== item 2) {continue}
  set sum (+ sum item)
 }
 for item in [Item {value:5},Item {value:6}] {set sum (+ sum item.value) break}
 assert (== sum 9)
 assert (== item 99)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            entries=sorted((b for b in bindings if b[0]=='main' and b[4]=='item'),key=lambda b:b[2])
            self.assertEqual(len(entries),3)
            self.assertEqual(len({b[1] for b in entries}),3)
            self.assertLessEqual(entries[1][3],entries[2][2])
            self.assertGreaterEqual(entries[0][3],entries[2][3])
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['item','item','item','sum'])
        self.produce(source,0,inspect)

    def test_match_arm_names_include_guard_and_exclude_other_arms(self):
        source='''union Box {Some {value:int} None {}}
fn read(box:Box)->int {
 let item:int = 99
 let result:int = match box {
  Some(item) if (< item.value 0) => 0
  Some(item) => item.value
  None(_) => 1
 }
 assert (== item 99)
 return result
}
shadow read {assert (== (read Box.Some {value:7}) 7) assert (== (read Box.Some {value:-1}) 0)}
fn main()->int {assert (== (read Box.Some {value:7}) 7) assert (== (read Box.None {}) 1) return 0}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            entries=sorted((b for b in bindings if b[0]=='read' and b[4]=='item'),key=lambda b:b[2])
            self.assertEqual(len(entries),3)
            self.assertEqual(len({b[1] for b in entries}),3)
            self.assertLessEqual(entries[1][3],entries[2][2])
            self.assertGreaterEqual(entries[0][3],entries[2][3])
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='read'),['box','item','item','item','result'])
        main_at=source.index('fn main()')
        read_at=source.index('fn read(')
        caller_first=source[:read_at]+source[main_at:]+source[read_at:main_at]
        for order,program in [('callee-first',source),('caller-first',caller_first)]:
            with self.subTest(order=order):
                self.produce(program,0,inspect)

    def test_named_nested_function_binding_and_captures(self):
        source = """fn main()->int {
 let offset:int=4
 fn helper(value:int)->int {let answer:int=(+ value offset) return answer}
 return (helper 3)
}
shadow main {assert (== (main) 7)}
"""
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['helper','offset'])
            self.assertEqual(sorted(b[4] for b in bindings if b[0]!='main' and not b[0].startswith('__nanoisa_shadow_')),['answer','value'])
        def captures(bindings):
            self.assertEqual([b[4] for b in bindings],['offset'])
        self.produce(source,7,inspect,inspect_captures=captures)

    def test_named_recursive_capture_preserves_environment(self):
        source = """fn main()->int {
 let offset:int=1
 fn sum(value:int)->int {if (== value 0) {return offset} return (+ value (sum (- value 1)))}
 return (sum 3)
}
shadow main {assert (== (main) 7)}
"""
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['offset','sum'])
        self.produce(source,7,inspect)

    def test_legacy_par_let_names_effects_and_enclosing_scope(self):
        source='''let mut trace:int = 0
fn mark(value:int)->int {set trace (+ (* trace 10) value) return value}
shadow mark {assert true}
fn finish(left:int,right:int)->int {assert (== (+ left right) 3) set trace (+ (* trace 10) 3) return 0}
shadow finish {assert true}
fn main()->int {
 set trace 0
 let _:int = 99
 par-let _=(mark 1) other=(mark 2) in (finish _ other)
 assert (== trace 123)
 assert (== _ 1)
 assert (== other 2)
 par-let values=[7,9] word="kept" in (array_length values)
 assert (== (array_length values) 2)
 assert (== word "kept")
 let original:int=6
 par-let original=(+ original 1) in original
 assert (== original 7)
 return 0
}
shadow main {assert (== (main) 0)}
'''
        def inspect(bindings):
            self.assertEqual(sorted(b[4] for b in bindings if b[0]=='main'),['_','_','original','original','other','values','word'])
        self.produce(source,0,inspect)

    def test_legacy_par_let_discards_expression_results(self):
        for body in ['1.5','"kept"','[7,9]','(7,true)','Item {value:7}']:
            with self.subTest(body=body):
                source='struct Item {value:int}\nfn main()->int {par-let value=7 in '+body+' return value}\nshadow main {assert (== (main) 7)}\n'
                def inspect(bindings):
                    self.assertEqual([b[4] for b in bindings if b[0]=='main'],['value'])
                self.produce(source,7,inspect)

if __name__=="__main__": unittest.main()
