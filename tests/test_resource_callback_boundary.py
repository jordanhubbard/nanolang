"""I keep generic callback ownership behind a tested boundary."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_CALLBACK_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = '''resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
union Marker<T> { Mark { value: int } }
'''

class ResourceCallbackBoundary(unittest.TestCase):
    def check(self, body, reject=True):
        source = PRELUDE + body + '\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-resource-callback-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    self.assertIn('generic resource callback signatures need ownership lowering', diagnostic)
                    self.assertNotIn('C compilation failed', diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generic_result_parameter(self):
        self.check('fn probe(make: fn() -> Box<Handle>) -> int { return 0 }\nshadow probe { assert true }')

    def test_generic_parameter(self):
        self.check('fn probe(consume: fn(Box<Handle>) -> int) -> int { return 0 }\nshadow probe { assert true }')

    def test_nested_returned_callback(self):
        self.check('fn probe(make: fn() -> fn() -> Box<Handle>) -> int { return 0 }\nshadow probe { assert true }')

    def test_resource_collection(self):
        self.check('fn probe(make: fn() -> array<Handle>) -> int { return 0 }\nshadow probe { assert true }')

    def test_local_callback(self):
        self.check('''fn make() -> Box<Handle> { return Box.None {} }
shadow make { assert true }
fn probe() -> int { let factory: fn() -> Box<Handle> = make return 0 }
shadow probe { assert true }''')

    def test_inferred_local_callback(self):
        self.check("""fn make() -> Box<Handle> { return Box.None {} }
shadow make { assert true }
fn probe() -> int { let factory = make return 0 }
shadow probe { assert true }""")

    def test_global_callback_signature(self):
        self.check("""fn make() -> Box<Handle> { return Box.None {} }
shadow make { assert true }
let factory: fn() -> Box<Handle> = make""")

    def test_returned_callback(self):
        self.check('''fn make() -> Box<Handle> { return Box.None {} }
shadow make { assert true }
fn probe() -> fn() -> Box<Handle> { return make }
shadow probe { assert true }''')

    def test_forwarded_callback(self):
        self.check('''fn sink(make: fn() -> Box<Handle>) -> int { return 0 }
shadow sink { assert true }
fn probe(make: fn() -> Box<Handle>) -> int { return (sink make) }
shadow probe { assert true }''')

    def test_fixed_resource_result(self):
        self.check("""fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn make() -> Handle { return Handle { fd: 7 } }
shadow make { assert (== (close_handle (make)) 7) }
fn probe(factory: fn() -> Handle) -> int { let owner: Handle = (factory) return (close_handle owner) }
shadow probe { assert (== (probe make) 7) }""", False)

    def test_fixed_resource_parameter(self):
        self.check("""fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn probe(consume: fn(Handle) -> int) -> int { let owner: Handle = Handle { fd: 7 } return (consume owner) }
shadow probe { assert (== (probe close_handle) 7) }""", False)

    def test_fixed_resource_void_callback(self):
        self.check("""fn discard(owner: Handle) -> void { let Handle { fd } = owner assert (== fd 7) }
shadow discard { (discard Handle { fd: 7 }) }
fn probe(drop: fn(Handle) -> void) -> int {
    let owner: Handle = Handle { fd: 7 }
    (drop owner)
    return 7
}
shadow probe { assert (== (probe discard) 7) }""", False)

    def test_fixed_resource_returned_callback_alias(self):
        self.check("""fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn close_other(owner: Handle) -> int { let Handle { fd } = owner return (+ fd 1) }
shadow close_other { assert (== (close_other Handle { fd: 7 }) 8) }
fn choose(flag: bool) -> fn(Handle) -> int { if flag { return close_handle } return close_other }
shadow choose { let consume: fn(Handle) -> int = (choose true) assert (== (consume Handle { fd: 7 }) 7) }
fn probe(flag: bool) -> int {
    let consume: fn(Handle) -> int = (choose flag)
    let alias: fn(Handle) -> int = consume
    let owner: Handle = Handle { fd: 7 }
    return (alias owner)
}
shadow probe { assert (== (probe true) 7) assert (== (probe false) 8) }""", False)

    def test_ordinary_generic_callback(self):
        self.check('''fn make() -> Box<int> { return Box.Some { value: 7 } }
shadow make { let value: Box<int> = (make) match value { Some(payload) => { assert (== payload.value 7) }, None(empty) => { assert false } } }
fn probe(make_value: fn() -> Box<int>) -> int { let value: Box<int> = (make_value) match value { Some(payload) => { return payload.value }, None(empty) => { return 0 } } }
shadow probe { assert (== (probe make) 7) }''', False)

    def test_phantom_resource_callback(self):
        self.check('''fn make() -> Marker<Handle> { return Marker.Mark { value: 7 } }
shadow make { let value: Marker<Handle> = (make) match value { Mark(payload) => { assert (== payload.value 7) } } }
fn probe(make_value: fn() -> Marker<Handle>) -> int { let value: Marker<Handle> = (make_value) match value { Mark(payload) => { return payload.value } } }
shadow probe { assert (== (probe make) 7) }''', False)

if __name__ == '__main__':
    unittest.main()
