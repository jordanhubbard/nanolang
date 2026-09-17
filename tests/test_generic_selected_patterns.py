"""I preserve concrete arguments through complete selected-variant patterns."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_affine_generic_identity as generic


class GenericSelectedPatterns(unittest.TestCase):
    def check(self, source, accepted):
        for compiler in generic.COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-generic-pattern-') as tmp:
                root = Path(tmp)
                program, output = root / 'main.nano', root / 'program'
                program.write_text(source)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(generic.COMPILER_ROOT / compiler), str(program), '-o', str(output)], cwd=generic.ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                if accepted:
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def ordinary(self, pattern, payload='int', literal='7', body='return value', expected='7'):
        return f'''union Box<T> {{ Some {{ value: T }}, Other {{ value: T }}, None {{}} }}
fn read(boxed: Box<{payload}>) -> int {{ match boxed {{
 Some(payload) => {{ {pattern} {body} }}
 Other(payload) => {{ return 0 }}
 None(payload) => {{ let Box.None {{}} = payload return 0 }}
}} }}
shadow read {{ let boxed: Box<{payload}> = Box.Some {{ value: {literal} }} assert (== (read boxed) {expected}) let empty: Box<{payload}> = Box.None {{}} assert (== (read empty) 0) }}
fn main() -> int {{ let boxed: Box<{payload}> = Box.Some {{ value: {literal} }} return (- (read boxed) {expected}) }}
shadow main {{ assert (== (main) 0) }}
'''

    def test_integer(self):
        self.check(self.ordinary('let Box.Some { value } = payload'), True)

    def test_string(self):
        self.check(self.ordinary('let Box.Some { value } = payload', 'string', '"kept"', 'assert (== value "kept") return 7'), True)

    def test_missing(self):
        self.check(self.ordinary('let Box.Some {} = payload', body='return 0'), False)

    def test_duplicate(self):
        self.check(self.ordinary('let Box.Some { value, value } = payload'), False)

    def test_wrong_variant(self):
        self.check(self.ordinary('let Box.Other { value } = payload'), False)

    def test_substituted_type_mismatch(self):
        self.check(self.ordinary('let Box.Some { value } = payload', 'string', '"kept"'), False)

    def test_owned_generic_transfer(self):
        generic.GenericAffineIdentity().check("""resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
fn consume(boxed: Box<Handle>) -> int { match boxed {
 Some(payload) => { let Box.Some { value } = payload return (close_handle value) }
 None(payload) => { return 0 }
} }
fn main() -> int { let owner: Handle = Handle { fd: 7 } let boxed: Box<Handle> = Box.Some { value: owner } return (- (consume boxed) 7) }
shadow main { assert (== (main) 0) }
""", True)

    def test_nested(self):
        source = '''union Result<T, E> { Ok { value: T }, Err { error: E } }
union Box<T> { Some { value: T }, None {} }
fn read(boxed: Box<Result<int, string>>) -> int { match boxed {
 Some(payload) => { let Box.Some { value } = payload match value {
  Ok(result) => { let Result.Ok { value } = result return value }
  Err(result) => { let Result.Err { error } = result assert (== error "failed") return 0 }
 } }
 None(payload) => { let Box.None {} = payload return 0 }
} }
shadow read { let result: Result<int, string> = Result.Ok { value: 7 } let boxed: Box<Result<int, string>> = Box.Some { value: result } assert (== (read boxed) 7) let failure: Result<int, string> = Result.Err { error: "failed" } let failed_box: Box<Result<int, string>> = Box.Some { value: failure } assert (== (read failed_box) 0) }
fn main() -> int { let result: Result<int, string> = Result.Ok { value: 7 } let boxed: Box<Result<int, string>> = Box.Some { value: result } return (- (read boxed) 7) }
shadow main { assert (== (main) 0) }
'''
        self.check(source, True)

if __name__ == '__main__':
    unittest.main()
