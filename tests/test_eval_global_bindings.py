"""I retain checked global identity across caller-local collisions and producers."""
from pathlib import Path
import gzip
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


class EvalGlobalBindings(unittest.TestCase):
    def test_original_witness_and_runtime_collisions(self):
        with tempfile.TemporaryDirectory(prefix='nano-global-bindings-') as name:
            directory = Path(name)
            original = directory/'original.nano'
            original.write_bytes(gzip.decompress((ROOT/'docs/evidence/nanolang-instructions-20261120/instructions-caller-global.nano.gz').read_bytes()))
            for fixture, expected, subdirectory in ((original, '', 'original'),
                    (ROOT/'tests/eval_global_bindings.nano', '42\n', 'runtime')):
                output = directory/subdirectory
                output.mkdir()
                routes = build_routes(fixture, output)
                for route, command in routes:
                    with self.subTest(fixture=fixture.name, route=route):
                        self.assertEqual(checked(command), expected)
            print(f'I retained the unchanged scope witness and runtime global collisions across {len(routes)} routes.')

    def test_qualified_global_storage_and_callee_order(self):
        with tempfile.TemporaryDirectory(prefix='nano-global-module-') as name:
            directory = Path(name)
            (directory/'owner.nano').write_text('''module owned_globals
pub let mut count: int = 41
pub let mut text: string = "global"
fn add_one(value: int) -> int { return (+ value 1) }
shadow add_one { assert (== (add_one 1) 2) }
fn add_many(value: int) -> int { return (+ value 100) }
shadow add_many { assert (== (add_many 1) 101) }
pub let mut apply: fn(int) -> int = add_one
pub fn swap() -> int { set apply add_many return 1 }
shadow swap { set apply add_one assert (== (swap) 1) assert (== (apply 1) 101) }
pub fn reset() -> void { set count 41 set text "global" set apply add_one }
shadow reset { (reset) assert (== count 41) assert (== text "global") assert (== (apply 1) 2) }
''')
            fixture = directory/'main.nano'
            fixture.write_text('''module "owner.nano" as globals
fn read_and_write() -> int {
    set globals.count (+ globals.count 1)
    assert (== globals.text "global")
    set globals.text "changed"
    let retained: fn(int) -> int = globals.apply
    let answer: int = (globals.apply (globals.swap))
    assert (== (retained 1) 2)
    assert (== answer 2)
    assert (== (globals.apply 1) 101)
    assert (== globals.text "changed")
    return globals.count
}
shadow read_and_write { (globals.reset) assert (== (read_and_write) 42) }
fn caller() -> int {
    (globals.reset)
    let count: int = -7
    let text: string = "caller"
    let answer: int = (read_and_write)
    assert (== count -7)
    assert (== text "caller")
    return answer
}
shadow caller { assert (== (caller) 42) }
fn main() -> int { (println (caller)) return 0 }
shadow main { assert (== (main) 0) }
''')
            output = directory/'products'
            output.mkdir()
            # I retain the legacy AST-to-C refusal under #986 separately.
            routes = build_routes(fixture, output, include_seed_native=False)
            for route, command in routes:
                with self.subTest(route=route):
                    self.assertEqual(checked(command), '42\n')
            print(f'I retained qualified global storage and callee-before-argument order across {len(routes)} routes.')
