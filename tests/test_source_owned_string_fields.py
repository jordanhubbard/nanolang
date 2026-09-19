"""I preserve the original STRING Bundle while qualifying exact retained fields."""
import ast
from pathlib import Path
import subprocess
import unittest
from tests import test_source_borrow_emission as support
from tests.test_owned_record_patterns import PREFIX
ROOT = Path(__file__).resolve().parents[1]

class SourceOwnedStringFields(support.SourceBorrowEmission):
    @classmethod
    def tearDownClass(cls):
        cls.temporary._finalizer.detach()
        print(f'I retain STRING field source fixtures at {cls.work}')

    def test_unchanged_original_bundle_and_source_routes(self):
        tree = ast.parse((ROOT/'tests/test_owned_record_patterns.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'test_nested_transfer_and_unsafe_scope')
        call = method.body[0].value
        self.assertIsInstance(call, ast.Call)
        text = PREFIX + ast.literal_eval(call.args[0])
        normal, shadow = self.graph_positive('original-string-bundle', text, b'', b'')
        self.assertIn('OWN_PACK 1', normal)
        self.assertIn('OWN_UNPACK_LOCAL', shadow)
        self.assertIn('EQ', normal)
        source = self.work/'original-string-bundle.nano'
        source.write_text(text)
        self.command(ROOT/'bin/nano', source)
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            binary = self.work/(compiler+'-original-string-bundle')
            self.command(ROOT/'bin'/compiler, source, '-o', binary)
            self.assertEqual(self.command(binary).stdout, '')

    def test_fields_results_aliases_and_order(self):
        text = '''resource struct Handle { fd: int }
struct Bundle { file: Handle, label: string }
fn close(owned: Handle) -> int { let Handle { fd } = owned return fd }
shadow close { assert (== (close Handle { fd: 7 }) 7) }
fn number(value: int, text: string) -> int { (print text) return value }
shadow number { assert true }
fn factory(first: bool, label: string) -> Bundle {
    if first { return Bundle { label: label, file: Handle { fd: (number 19 "A") } } }
    else { return Bundle { file: Handle { fd: (number 19 "B") }, label: label } }
}
shadow factory { assert true }
fn relay(bundle: Bundle) -> Bundle { return bundle }
shadow relay { assert true }
fn consume(bundle: Bundle) -> int {
    let before: string = bundle.label
    let Bundle { label, file } = bundle
    let alias: string = label
    let mut changed: string = label
    set changed "changed"
    assert (== alias before)
    assert (!= changed alias)
    return (close file)
}
shadow consume { assert true }
fn main() -> int {
    assert (== (consume (relay (factory true "ready"))) 19)
    assert (== (consume (relay (factory false ""))) 19)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        self.graph_positive('string-field-results', text, b'AB', b'AB')
        local = self.owned_string_fixture().replace('(print "")', 'let extra: string = "extra" (print extra)')
        expected = b'A\tB\rC\n"\'\\0\\q caf\xc3\xa9\nextra'
        self.graph_positive('new-string-local', local, expected, expected)
        nested = self.transitive_wrapper_fixture().replace('extra: int', 'extra: string') \
            .replace('extra: (scalar 9 "B")', 'extra: "B"').replace('return (+ value extra)', 'return value') \
            .replace('(take owner) 16', '(take owner) 7').replace('extra: "B" }) 16', 'extra: "B" }) 7')
        self.graph_positive('new-transitive-string', nested, b'AA', b'AA')

    def test_retained_refusals_preserve_output(self):
        base = '''resource struct Handle { fd: int }
struct Bundle { file: Handle, label: string }
fn consume(bundle: Bundle) -> int {
    let Bundle { file, label } = bundle
    let Handle { fd } = file
    assert (== label "ready")
    return fd
}
shadow consume { assert true }
fn main() -> int { assert (== (consume Bundle { file: Handle { fd: 7 }, label: "ready" }) 7) return 0 }
shadow main { assert (== (main) 0) }
'''
        cases = {
            'order': base.replace('(== label "ready")', '(< label "ready")'),
            'concat': base.replace('(== label "ready")', '(== (+ label "!") "ready!")'),
            'result': base.replace('-> int {\n    let Bundle', '-> string {\n    let Bundle').replace('return fd', 'return label'),
            'float_field': base.replace('label: string', 'label: float').replace('label: "ready"', 'label: 1.5').replace('(== label "ready")', '(== label 1.5)'),
            'borrowed': '''resource struct Text { label: string }
fn read(text: &Text) -> int { assert (== text.label "ready") return 0 }
shadow read { assert true }
fn main() -> int { let text: Text = Text { label: "ready" } assert (== (read &text) 0) let Text { label } = text assert (== label "ready") return 0 }
shadow main { assert (== (main) 0) }
''',
        }
        for name, text in cases.items():
            source = self.work/f'string-field-refused-{name}.nano';source.write_text(text)
            for compiler in [ROOT/'bin'/x for x in ('nano_virt','nanoc_stage1','nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler.name):
                    output=self.work/'retained-prior.nvm';output.write_bytes(b'prior verified artifact')
                    args=[compiler,source]
                    if compiler not in self.emitters:args.append('--emit-nvm')
                    run=subprocess.run([*args,'-o',output],cwd=ROOT,capture_output=True,text=True,timeout=180)
                    self.assertGreater(run.returncode,0,run.stdout+run.stderr)
                    self.assertEqual(output.read_bytes(),b'prior verified artifact')
                    self.assertNotRegex(run.stdout+run.stderr,r'(?i)parse (?:error|failed)|unexpected token')


def load_tests(loader, standard_tests, pattern):
    return unittest.TestSuite(SourceOwnedStringFields(name) for name in (
        'test_unchanged_original_bundle_and_source_routes',
        'test_fields_results_aliases_and_order',
        'test_retained_refusals_preserve_output'))
