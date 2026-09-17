"""I retain borrow syntax and guard annotation contexts without lowering support."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_BORROW_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = 'resource struct Handle { fd: int }\n'
MAIN = '\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'

class BorrowAnnotations(unittest.TestCase):
    def compile(self, source, message=None):
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-borrow-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                diagnostic = result.stdout + result.stderr
                if message:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    self.assertIn(message, diagnostic)
                    self.assertNotIn('C compilation failed', diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    ran = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_parameters_remain_guarded(self):
        for annotation in ('&mut Handle', '&array<int>', '&mut Box<int>'):
            with self.subTest(annotation=annotation):
                self.compile(PRELUDE + 'union Box<T> { Value { value: T } }\n'
                             + f'fn observe(value: {annotation}) -> int {{ return 0 }}\n'
                             + 'shadow observe { assert true }\n' + MAIN,
                             'borrow')

    def test_extern_parameter_cannot_bypass_guard(self):
        self.compile(PRELUDE + 'extern fn observe(value: &Handle) -> int\n' + MAIN,
                     'borrow')

    def test_unsupported_annotation_contexts(self):
        for declaration in ('extern fn value() -> &Handle',
                            'struct Stored { value: &Handle }',
                            'extern fn nested(value: &&Handle) -> int',
                            'extern fn callback(value: fn(&Handle)->int) -> int',
                            'extern fn collection(value: array<&Handle>) -> int'):
            with self.subTest(declaration=declaration):
                self.compile(PRELUDE + declaration + MAIN,
                             'borrow annotations only on named parameters')

    def test_arguments_are_explicitly_unsupported(self):
        self.compile('fn observe(value: int) -> int { return value }\nshadow observe { assert true }\n'
                     'fn main() -> int { let value: int = 1 return (observe &value) }\nshadow main { assert true }\n',
                     'borrow')

    def test_ordinary_ownership_control(self):
        self.compile(PRELUDE + 'fn consume(value: Handle) -> int { let Handle { fd } = value return fd }\n'
                     'shadow consume { assert (== (consume Handle { fd: 7 }) 7) }\n'
                     'fn main() -> int { assert (== (consume Handle { fd: 7 }) 7) return 0 }\n'
                     'shadow main { assert (== (main) 0) }\n')

    def test_selfhost_parser_retains_annotation_identity(self):
        self.compile('''import "src_nano/compiler/lexer.nano"
import "src_nano/parser.nano"
fn main() -> int {
    let tokens: List<LexerToken> = (tokenize_string "extern fn inspect(a: &owned.Handle, b: &mut Box<array<int>>) -> int" "borrow.nano" (list_CompilerDiagnostic_new))
    let parsed: Parser = (parse_program tokens (list_LexerToken_length tokens) "borrow.nano")
    assert (not (parser_has_error parsed))
    let function: ASTFunction = (parser_get_function parsed 0)
    assert (== function.param_count 2)
    let a: ASTLet = (parser_get_let parsed function.param_start)
    let b: ASTLet = (parser_get_let parsed (+ function.param_start 1))
    assert (== a.var_type "&owned.Handle")
    assert (== b.var_type "&mut Box<array<int>>")
    return 0
}
shadow main { assert (== (main) 0) }
''')

if __name__ == '__main__':
    unittest.main()
