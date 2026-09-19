"""I retain the complete Bundle/PREFIX while qualifying my distinct owner-array route."""
import ast
import subprocess
import unittest
from tests import test_source_mixed_samples as support
from tests.test_owned_record_patterns import PREFIX, OwnedRecordPatterns

ROOT = support.ROOT


class SourceOwnedFloatArrays(support.SourceMixedSamples):
    @staticmethod
    def original():
        tree = ast.parse((ROOT / 'tests/test_owned_record_patterns.py').read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == 'test_ordinary_array_field_keeps_element_type')
        return PREFIX + ast.literal_eval(method.body[0].value.args[0])

    def refused(self, name, text, shadows=False):
        source = self.work / (name + '.nano')
        source.write_text(text)
        compilers = [ROOT / 'bin' / item for item in
                     ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')]
        if not shadows:
            compilers += self.emitters
        for compiler in compilers:
            with self.subTest(case=name, compiler=compiler.name):
                output = self.work / (name + '-' + compiler.name + '.prior')
                output.write_bytes(b'previous verified publication')
                args = [compiler, source, '-o', output]
                if compiler not in self.emitters:
                    args.append('--emit-nvm')
                result = subprocess.run(args, cwd=ROOT, capture_output=True,
                                        text=True, timeout=180)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'previous verified publication')
                self.assertNotRegex(result.stdout + result.stderr,
                                    r'(?i)parse (?:error|failed)|unexpected token')
                self.assertRegex(result.stdout + result.stderr,
                                 r'(?i)type|array|owner|require|unsupported|shadow|assert|call|builtin|field|verify')

    def test_original_complete_prefix_and_all_routes(self):
        normal, shadows = self.graph_positive('owned-array-original', self.original(), b'', b'')
        self.assertIn('OWN_PACK 1', normal)
        self.assertIn('OWN_UNPACK_LOCAL', normal)
        self.assertIn('ARR_GET', normal)
        self.assertNotIn('AGG_PACK', normal)
        self.assertNotIn('REF_GET', shadows)
        OwnedRecordPatterns('test_ordinary_array_field_keeps_element_type').test_ordinary_array_field_keeps_element_type()

    def test_integer_boundaries_and_boolean_range(self):
        checks = '''let minimum: int = (- (- 0 9223372036854775807) 1)
    assert (== (+ 9223372036854775807 1) minimum)
    assert (== (- minimum 1) 9223372036854775807)
    assert (== (* minimum -1) minimum)
    assert (== (- minimum) minimum)
    assert (== (/ minimum -1) minimum) assert (== (% minimum -1) 0)
    assert (== (/ 7 0) 0) assert (== (% 7 0) 0)
    assert (and true (not false)) assert (or false true)
    let mut total: int = 0
    for index in (range 0 3) { set total (+ total index) }
    assert (== total 3)
    while (< total 5) { set total (+ total 1) }
    assert (== total 5)
    '''
        text = self.original().replace('    return 0', '    ' + checks + 'return 0')
        normal, _ = self.graph_positive('owned-array-integers', text, b'', b'')
        for opcode in ('I64_ADD', 'I64_SUB', 'I64_MUL', 'I64_DIV_S', 'I64_REM_S', 'I64_NEG', 'BOOL_NOT'):
            self.assertIn(opcode, normal)

    def test_prepared_roots_nested_returns_alias_and_empty(self):
        text = PREFIX + '''struct Bundle { file: Handle, samples: array<float> }
fn make(value: int) -> Handle { return Handle { fd: value } }
shadow make { assert (== (close (make 11)) 11) }
fn forward(bundle: Bundle) -> Bundle { return bundle }
shadow forward {
    let returned: Bundle = (forward Bundle { file: Handle { fd: 13 }, samples: [3.5] })
    assert (== returned.file.fd 13)
    let Bundle { samples, file } = returned
    assert (== (at samples 0) 3.5)
    assert (== (close file) 13)
}
fn main() -> int {
    let empty: Bundle = Bundle { samples: [], file: (make 3) }
    let Bundle { file, samples } = empty
    assert (== (close file) 3)
    let bundle: Bundle = (forward Bundle { samples: [1.5, 2.5], file: (make 7) })
    let before = bundle.samples
    assert (== bundle.file.fd 7)
    let Bundle { samples, file } = bundle
    let alias = before
    assert (== (at alias 0) (at samples 0))
    assert (== (close file) 7)
    assert (== (at alias 1) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        normal, _ = self.graph_positive('owned-array-roots', text, b'', b'')
        self.assertLess(normal.index('ARR_LITERAL 3 2'), normal.rindex('PUSH_I64 7'))

    def test_optional_operands_and_failure_cleanup(self):
        base = self.original()
        for name, expression in (
                ('left', '(== (at samples 0) 1.5)'),
                ('right', '(== 1.5 (at samples 0))'),
                ('both', '(== (at samples 0) (at samples 0))')):
            self.graph_positive('optional-valid-' + name,
                                base.replace('(== (at samples 0) 1.5)', expression), b'', b'')
            failed = base.replace('(== (at samples 0) 1.5)', expression.replace('samples 0', 'samples 99'))
            # My public compilers must run every original shadow before publication.
            self.refused('optional-shadow-' + name, failed, shadows=True)
            # My raw emitters expose checked runtime failure; no rejected module executes.
            source = self.work / ('optional-runtime-' + name + '.nano')
            source.write_text(failed)
            for emitter in self.emitters:
                assembly = self.work / ('optional-' + name + '-' + emitter.name + '.nasm')
                module = assembly.with_suffix('.nvm')
                self.command(emitter, source, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.execute_pair(module, expected=1, expected_output=b'')

    def test_exact_type_lexical_and_shadow_refusals(self):
        base = self.original()
        cases = {
            'optional-local': base.replace('    return 0', '    let exact: float = (at samples 0) return 0'),
            'optional-add': base.replace('(== (at samples 0) 1.5)', '(== (+ (at samples 0) 1.0) 2.5)'),
            'optional-negate': base.replace('(== (at samples 0) 1.5)', '(== (- (at samples 0)) -1.5)'),
            'bound-at': base.replace('    assert (== (at samples 0)', '    let at: int = 7 assert (== (at samples 0)'),
            'bound-not': base.replace('    return 0', '    let not: int = 7 assert (not false) return 0'),
            'wrong-element': base.replace('[1.5, 2.5]', '[1, 2]'),
            'untyped-empty': base.replace('    return 0', '    let unknown = [] return 0'),
            'duplicate-owner': base.replace('    return 0', '    (close file) return 0'),
            'dead-parent': base.replace('    return 0', '    let stale = bundle.samples return 0'),
            'missing-pattern': base.replace('let Bundle { samples, file }', 'let Bundle { file }'),
            'false-close': base.replace('(close Handle { fd: 7 }) 7', '(close Handle { fd: 7 }) 8'),
            'false-main': base.replace('(main) 0', '(main) 1'),
        }
        for name, text in cases.items():
            self.refused(name, text, shadows=name.startswith('false-'))

    def test_unchanged_samples_string_scalar_and_borrow_profiles(self):
        for method in ('test_ordinary_inferred_field_and_alias',
                       'test_nested_transfer_and_unsafe_scope',
                       'test_scalar_terminal_operation',
                       'test_unsafe_pattern_keeps_outer_shadow'):
            original = OwnedRecordPatterns(method)
            getattr(original, method)()
        self.test_owned_string_unchanged_affine_example()
        source = (support.support.FIXTURES / 'source_borrow_shared.nano').read_text()
        self.graph_positive('unchanged-borrow', source)


def load_tests(loader, standard_tests, pattern):
    # I deliberately exclude inherited suites from this bounded qualification.
    names = [name for name in SourceOwnedFloatArrays.__dict__ if name.startswith('test_')]
    return unittest.TestSuite(SourceOwnedFloatArrays(name) for name in names)
