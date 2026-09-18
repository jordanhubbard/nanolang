"""I retain runtime slots while restoring lexical match-arm bindings."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MatchBindings(unittest.TestCase):
    def check(self, body):
        source = '''union Item { Some { n:int }, None {} }
fn main()->int {
''' + body + '''
 return 0
}
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory(prefix='nano-match-bindings-') as tmp:
            path = Path(tmp)/'ordinary.nano'
            module = Path(tmp)/'ordinary.nvm'
            path.write_text(source)
            for command in ([ROOT/'bin/nano', path],
                            [ROOT/'bin/nano_virt', path, '--emit-nvm', '-o', module],
                            [ROOT/'bin/nano_vm', '--verify-only', module],
                            [ROOT/'bin/nano_vm', module]):
                result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0,
                                 str(command)+'\n'+result.stdout+result.stderr)

    def test_nested_payload_after_closed_loop_scope(self):
        self.check('''let mut index:int=0
 while (< index 2) { let temporary:int=index set index (+ temporary 1) }
 let payload:int=19
 match Item.Some{n:31} {
  Some(payload) if (> payload.n 0) => {
   match Item.Some{n:47} {
    Some(payload) if (> payload.n 0) => { assert (== payload.n 47) }
    _ => { assert false }
   }
   assert (== payload.n 31)
  }
  _ => { assert false }
 }
 assert (== payload 19)
''')

    def test_guard_false_sibling_and_discard(self):
        self.check('''let mut index:int=0
 while (< index 1) { let temporary:int=1 set index temporary }
 let payload:int=23 let _:int=29
 match Item.Some{n:41} {
  Some(payload) if (< payload.n 0) => { assert false }
  Some(payload) if (== payload.n 41) => { let local:int=payload.n assert (== local 41) }
  _ => { assert false }
 }
 assert (== payload 23)
 match Item.Some{n:53} { Some(_) => { assert (== _ 29) } _ => { assert false } }
 assert (== payload 23)
''')

    def test_expression_and_statement_binding_scopes(self):
        self.check('''let mut index:int=0
 while (< index 1) { let temporary:int=1 set index temporary }
 let payload:int=13
 let result:int=match Item.Some{n:17} {
  Some(payload) => (+ (match Item.Some{n:7} { Some(payload) => payload.n None(empty) => 0 }) payload.n)
  None(empty) => 0
 }
 assert (== result 24) assert (== payload 13)
''')

if __name__ == '__main__':
    unittest.main()
