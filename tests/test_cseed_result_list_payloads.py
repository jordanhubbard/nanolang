"""I preserve checked Result payloads and generated List names in native output."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CSeedResultListPayloads(unittest.TestCase):
    def compile(self, source, *, success=True):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-payload-") as temporary:
            directory = Path(temporary)
            path, output = directory / "input.nano", directory / "program"
            path.write_text(source)
            output.write_bytes(b"prior-output")
            compiler = os.environ.get("NANO_CSEED_PAYLOAD_COMPILER", str(ROOT / "bin/nanoc_c"))
            result = subprocess.run([compiler, path, "-o", output], cwd=ROOT,
                                    capture_output=True, timeout=30)
            if not success:
                self.assertNotEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(output.read_bytes(), b"prior-output")
                return
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertNotEqual(output.read_bytes(), b"prior-output")
            executed = subprocess.run([output], capture_output=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_declared_payload_names_tags_and_once_evaluation(self):
        self.compile('''
union Payload { Err { code: int }, Ok { text: string } }
let mut calls: int = 0
fn good() -> Payload { set calls (+ (* calls 10) 1) return Payload.Ok { text: "kept" } }
shadow good { assert (== (result_unwrap (good)) "kept") }
fn fallback() -> string { set calls (+ (* calls 10) 2) return "fallback" }
shadow fallback { assert (== (fallback) "fallback") }
fn main() -> int {
    set calls 0
    let text: string = (result_unwrap_or (good) (fallback))
    assert (== text "kept")
    assert (== calls 12)
    assert (== (result_unwrap (good)) "kept")
    let failure: Payload = Payload.Err { code: 37 }
    assert (== (result_unwrap_err failure) 37)
    assert (== (result_unwrap_or failure "fallback") "fallback")
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_conventional_generic_result_payloads(self):
        self.compile('''
union Result<T, E> { Ok { value: T }, Err { error: E } }
fn main() -> int {
    let good: Result<int, string> = Result.Ok { value: 42 }
    let bad: Result<int, string> = Result.Err { error: "failure" }
    assert (== (result_unwrap good) 42)
    assert (== (result_unwrap_err bad) "failure")
    assert (== (result_unwrap_or bad 7) 7)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_generated_list_operations_and_callable_shadow(self):
        self.compile('''
struct Item { value: int }
fn distinct(value: int) -> int { return (+ value 3) }
shadow distinct { assert (== (distinct 4) 7) }
fn List_Ordinary_new(value: int) -> int { return (+ value 2) }
shadow List_Ordinary_new { assert (== (List_Ordinary_new 5) 7) }
fn main() -> int {
    unsafe {
        let items: List<Item> = (List_Item_new)
        (List_Item_push items Item { value: 11 })
        (List_Item_push items Item { value: 13 })
        let picked: Item = (List_Item_get items 1)
        assert (== picked.value 13)
        assert (== (List_Item_length items) 2)
        let List_Item_length: fn(int) -> int = distinct
        assert (== (List_Item_length 4) 7)
        assert (== (List_Ordinary_new 5) 7)
    }
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_wrong_variant_unknown_operation_and_argument_preserve_output(self):
        sources = [
            '''union Payload { Ok { text: string }, Err { code: int } }
fn main() -> int { let value: Payload = Payload.Err { code: 7 } (result_unwrap value) return 0 }
shadow main { assert (== (main) 0) }
''',
            '''struct Item { value: int }
fn main() -> int { unsafe { let items: List<Item> = (List_Item_new) (List_Item_missing items) } return 0 }
shadow main { assert (== (main) 0) }
''',
            '''struct Item { value: int }
fn main() -> int { unsafe { let items: List<Item> = (List_Item_new) (List_Item_push items 7) } return 0 }
shadow main { assert (== (main) 0) }
''',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.compile(source, success=False)
