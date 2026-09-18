"""I retain lexical payload scopes for complete expression match arms."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'union Choice { Some { number: int }, None {} }\n'


class MatchExpressionScope(unittest.TestCase):
    def check(self, body, accepted=True):
        source = PREFIX + body + '''
fn main() -> int {
 assert (== (check Choice.Some { number: 7 }) 10)
 assert (== (check Choice.None {}) 3)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='match-expression-scope-') as tmp:
            source_path, output = Path(tmp)/'case.nano', Path(tmp)/'output'
            source_path.write_text(source)
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2', 'nano_virt'):
                with self.subTest(compiler=compiler):
                    output.write_bytes(b'prior output')
                    args = [str(ROOT/'bin'/compiler), str(source_path), '-o', str(output)]
                    if compiler == 'nano_virt':
                        args += ['--emit-nvm']
                    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=180)
                    if not accepted:
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior output')
                        continue
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
                    run_args = [str(output)] if compiler != 'nano_virt' else [str(ROOT/'bin/nano_vm'), str(output)]
                    run = subprocess.run(run_args, cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_same_line_outer_binding_and_sibling(self):
        self.check('''fn check(value: Choice) -> int { let payload: int = 3 let number: int = match value { Some(payload) => payload.number None(empty) => (- payload 3) } return (+ number payload) }
shadow check { assert (== (check Choice.Some { number: 7 }) 10) }
''')

    def test_multiline_outer_binding(self):
        self.check('''fn check(value: Choice) -> int {
 let payload: int = 3
 let number: int = match value {
  Some(payload) => (+ payload.number 0),
  None(payload) => 0
 }
 return (+ number payload)
}
shadow check { assert (== (check Choice.None {}) 3) }
''')

    def test_nested_match_restores_outer_payload(self):
        self.check('''fn check(value: Choice) -> int {
 let payload: int = 3
 let number: int = match value {
  Some(payload) => (+ (match Choice.Some { number: 0 } { Some(payload) => payload.number None(empty) => 0 }) payload.number)
  None(empty) => 0
 }
 return (+ number payload)
}
shadow check { assert (== (check Choice.Some { number: 7 }) 10) }
''')

    def test_payload_cannot_escape_expression(self):
        self.check('''fn check(value: Choice) -> int {
 let number: int = match value { Some(payload) => payload.number None(empty) => 0 }
 return (+ number payload.number)
}
shadow check { assert true }
''', False)


if __name__ == '__main__':
    unittest.main()
