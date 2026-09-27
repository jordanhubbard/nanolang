"""I preserve old bindings while publishing distinct native local storage."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''union Choice { Some { number: int }, None {} }
fn sum(left: int, right: int) -> int { return (+ left right) }
shadow sum { assert (== (sum 8 3) 11) }
fn difference(left: int, right: int) -> int { return (- left right) }
shadow difference { assert (== (difference 8 3) 5) }
fn invoke(callback: fn(int, int) -> int, expected: int) -> int {
 let callback: fn(int, int) -> int = callback
 let callback: fn(int, int) -> int = callback
 assert (== (callback 8 3) expected)
 if true {
  let mut callback: fn(int, int) -> int = callback
  assert (== (callback 8 3) expected)
  set callback difference
  assert (== (callback 8 3) 5)
 }
 assert (== (callback 8 3) expected)
 for callback in [2, 3] { assert (> callback 1) }
 assert (== (callback 8 3) expected)
 let selected: int = match Choice.Some { number: 7 } {
  Some(callback) => callback.number
  None(empty) => 0
 }
 assert (== selected 7)
 let guarded: int = match Choice.Some { number: 9 } {
  Some(callback) if (> callback.number 8) => callback.number
  Some(other) => 0
  None(empty) => 0
 }
 assert (== guarded 9)
 match Choice.Some { number: 7 } {
  Some(callback) => { assert (== callback.number 7) }
  None(empty) => { assert false }
 }
 match Choice.Some { number: 9 } {
  Some(callback) if (> callback.number 8) => { assert (== callback.number 9) }
  Some(other) => { assert false }
  None(empty) => { assert false }
 }
 return (callback 8 3)
}
shadow invoke { assert (== (invoke sum 11) 11) assert (== (invoke difference 5) 5) }
fn ordinary(value: int) -> int {
 let value: int = (+ value 1)
 let value: int = (+ value 2)
 if true { let value: int = (+ value 3) assert (== value 10) }
 return value
}
shadow ordinary { assert (== (ordinary 4) 7) }
fn main() -> int {
 assert (== (invoke sum 11) 11)
 assert (== (invoke difference 5) 5)
 assert (== (ordinary 4) 7)
 return 0
}
shadow main { assert (== (main) 0) }
'''

class CSeedBindingNames(unittest.TestCase):
    def test_aliases_scopes_and_runtime_targets(self):
        with tempfile.TemporaryDirectory(prefix='nano-binding-names-') as tmp:
            source, output = Path(tmp)/'case.nano', Path(tmp)/'program'
            source.write_text(SOURCE)
            for flags in ([], ['--tco']):
                with self.subTest(flags=flags):
                    for command in ([ROOT/'bin/nanoc_c', source, *flags, '-o', output], [output]):
                        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                                capture_output=True, text=True, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
