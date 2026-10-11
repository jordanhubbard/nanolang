"""I specialize every conditional callback branch from its declared context."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_contextual_generic_functions as shared
from tests import test_cseed_generic_functions as runner

PRELUDE = shared.PRELUDE + 'fn other(value:T)->T{return value}\nshadow other {assert (== (other 1) 1)}\n'

class ConditionalGenericCallbacks(unittest.TestCase):
    execute = shared.ContextualGenericFunctions.execute

    def test_local_nested_and_reassigned(self):
        self.execute(PRELUDE+'''fn main()->int {
 let mut number:fn(int)->int = (cond (true identity) (else other))
 assert (== (number 7) 7)
 set number (cond (false identity) (else other))
 assert (== (number 9) 9)
 let text:fn(string)->string = (cond (false identity) (else (cond (false other) (else identity))))
 assert (== (text "kept") "kept")
 return 0
}'''+shared.SHADOW)

    def test_return_and_call_argument(self):
        self.execute(PRELUDE+'''fn choose(flag:bool)->fn(int)->int {return (cond (flag identity) (else other))}
shadow choose {assert (== ((choose true) 7) 7) assert (== ((choose false) 9) 9)}
fn apply(callback:fn(int)->int)->int {return (callback 11)}
shadow apply {assert (== (apply identity) 11)}
fn main()->int {assert (== ((choose false) 9) 9) assert (== (apply (cond (false identity) (else other))) 11) return 0}
'''+shared.SHADOW)

    def test_record_array_and_union_storage(self):
        self.execute(PRELUDE+'''struct Holder {callback:fn(int)->int}
union Choice {Some {callback:fn(int)->int}}
fn read(choice:Choice)->int {match choice {Some(p)=>{let f:fn(int)->int = p.callback return (f 7)}}}
shadow read {assert (== (read Choice.Some {callback:identity}) 7)}
fn main()->int {
 let holder:Holder = Holder {callback:(cond (false identity) (else other))}
 let f:fn(int)->int = holder.callback
 assert (== (f 7) 7)
 let values:array<fn(int)->int> = [(cond (true identity) (else other))]
 let g:fn(int)->int = (at values 0)
 assert (== (g 9) 9)
 assert (== (read Choice.Some {callback:(cond (false identity) (else other))}) 7)
 return 0
}'''+shared.SHADOW)

    def test_conditions_execute_once_and_only_selected_callback_runs(self):
        self.execute(PRELUDE+'''let mut events:int = 0
fn condition(value:int)->bool {set events (+ (* events 10) value) return (== value 2)}
shadow condition {let saved:int = events assert (condition 2) set events saved}
fn increment(value:int)->int {return (+ value 1)}
shadow increment {assert (== (increment 7) 8)}
fn main()->int {
 set events 0
 let first:fn(int)->int = (cond ((condition 1) identity) ((condition 2) increment) (else identity))
 assert (== events 12)
 assert (== (first 7) 8)
 set events 0
 let second:fn(int)->int = (cond ((condition 2) identity) ((condition 1) increment) (else increment))
 assert (== events 2)
 assert (== (second 7) 7)
 return 0
}
shadow main {let saved:int = events assert (== (main) 0) set events saved}
''')

    def test_imported_branches_retain_callable_context(self):
        self.execute('''module "identity.nano" as values
from "identity.nano" import identity as same
fn main()->int {
 let first:fn(int)->int = (cond (true values.identity) (else same))
 let second:fn(string)->string = (cond (false same) (else values.identity))
 assert (== (first 7) 7)
 assert (== (second "kept") "kept")
 return 0
}'''+shared.SHADOW, files={'identity.nano':shared.PRELUDE.replace('fn identity', 'pub fn identity', 1)})

    def test_inactive_incompatible_branch_preserves_output(self):
        for value in ['(cond (true identity) (else wrong))',
                      '(cond (false wrong) (else identity))',
                      '(cond (true identity) (else (cond (false identity) (else wrong))))']:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'prior'
                source.write_text(PRELUDE+'fn wrong(x:int)->bool{return true}\nshadow wrong {assert (wrong 1)}\n'+
                                  'fn unused()->int{let f:fn(int)->int='+value+' return 0}\nshadow unused {assert true}\nfn main()->int{return 0}'+shared.SHADOW)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(runner.COMPILER), str(source), *runner.MODE, '-o', str(output)],
                                        cwd=runner.ROOT, capture_output=True, text=True, timeout=120)
                self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')
