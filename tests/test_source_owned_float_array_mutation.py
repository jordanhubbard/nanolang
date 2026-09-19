"""I qualify ordered mutation without changing owner signatures or array identity."""
import subprocess
import unittest
from tests import test_source_owned_float_arrays as source
from tests.test_owned_record_patterns import PREFIX

ROOT = source.ROOT


class SourceOwnedFloatArrayMutation(source.SourceOwnedFloatArrays):
    @staticmethod
    def mutation_fixture(body):
        return PREFIX + '''struct Bundle { file: Handle, left: array<float>, right: array<float> }
fn main() -> int {
    let values: array<float> = [0.5]
    let bundle: Bundle = Bundle { right: values, file: Handle { fd: 7 }, left: values }
''' + body + '''
    let Bundle { right, file, left } = bundle
    assert (== (close file) 7)
    return 0
}
shadow main { assert (== (main) 0) }
'''

    def test_shared_mutation_growth_append_result_and_empty(self):
        body = '''    let alias = bundle.left
    (array_set bundle.right 0 1.5)
    assert (== (at alias 0) 1.5)
    let appended = (array_push alias 2.5)
    (array_set appended 1 3.5)
    assert (== (at bundle.left 1) 3.5)
    (array_push bundle.right 4.5)
    assert (== (array_length values) 3)
    for index in (range 0 40) { (array_push alias 6.5) }
    assert (== (array_length appended) 43)
    assert (== (at bundle.right 42) 6.5)
    let empty: Bundle = Bundle { left: [], right: [], file: Handle { fd: 9 } }
    assert (== (array_length empty.left) 0)
    (array_push empty.left 7.5)
    assert (== (array_length empty.left) 1)
    assert (== (array_length empty.right) 0)
    let Bundle { file, right, left } = empty
    assert (== (at left 0) 7.5)
    assert (== (close file) 9)
'''
        text = self.mutation_fixture(body)
        normal, _ = self.graph_positive('mutation-shared-growth', text, b'', b'')
        self.assertIn('ARR_SET\n  POP', normal)
        self.assertIn('ARR_PUSH', normal)
        self.assertIn('ARR_LEN', normal)

    def test_nested_receiver_before_consuming_index(self):
        text = PREFIX + '''struct Bundle { file: Handle, values: array<float> }
fn index(owned: Bundle) -> int {
    let Bundle { values, file } = owned
    assert (== (array_length values) 2)
    assert (== (at values 1) 2.5)
    (array_set values 0 9.5)
    assert (== (close file) 7)
    return 1
}
shadow index { assert (== (index Bundle { file: Handle { fd: 7 }, values: [0.5, 2.5] }) 1) }
fn main() -> int {
    let values: array<float> = [0.5]
    let bundle: Bundle = Bundle { file: Handle { fd: 7 }, values: values }
    (array_set (array_push bundle.values 2.5) (index bundle) 3.5)
    assert (== (array_length values) 2)
    assert (== (at values 0) 9.5)
    assert (== (at values 1) 3.5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        self.graph_positive('mutation-consuming-index', text, b'', b'')

    def test_constructor_order_and_returned_aliases(self):
        text = PREFIX + '''struct Bundle { file: Handle, left: array<float>, right: array<float> }
fn make(length: int) -> Handle { assert (== length 2) return Handle { fd: 7 } }
shadow make { assert (== (close (make 2)) 7) }
fn forward(owned: Bundle) -> Bundle { return owned }
shadow forward {
    let returned: Bundle = (forward Bundle { file: Handle { fd: 7 }, left: [1.5], right: [2.5] })
    let Bundle { right, left, file } = returned
    assert (== (at left 0) 1.5) assert (== (at right 0) 2.5)
    assert (== (close file) 7)
}
fn main() -> int {
    let values: array<float> = [0.5]
    let returned: Bundle = (forward Bundle { right: (array_push values 1.5), file: (make (array_length values)), left: (array_push values 2.5) })
    assert (== (array_length returned.left) 3)
    assert (== (array_length returned.right) 3)
    let Bundle { right, file, left } = returned
    (array_set left 0 8.5)
    assert (== (at right 0) 8.5)
    assert (== (close file) 7)
    assert (== (at values 0) 8.5)
    assert (== (array_length (array_push values 3.5)) 4)
    assert (== (array_length values) 4)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        self.graph_positive('mutation-constructor-order', text, b'', b'')

    def test_builtin_identity_initializer_and_declared_function(self):
        text = self.mutation_fixture('''    let array_push = (array_push values 1.5)
    assert (== (array_length array_push) 2)
    if true { let array_length: int = 7 assert (== array_length 7) }
    assert (== (array_length values) 2)
''')
        self.graph_positive('mutation-initializer-scope', text, b'', b'')
        declared = PREFIX + '''struct Bundle { file: Handle, values: array<float> }
fn array_push(owner: Handle, expected: int) -> int { let Handle { fd } = owner assert (== fd expected) return fd }
shadow array_push { assert (== (array_push Handle { fd: 11 } 11) 11) }
fn main() -> int {
    let bundle: Bundle = Bundle { file: Handle { fd: 7 }, values: [1.5] }
    assert (== (array_push Handle { fd: 13 } 13) 13)
    assert (== (array_length bundle.values) 1)
    let Bundle { values, file } = bundle
    assert (== (close file) 7)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        normal, _ = self.graph_positive('mutation-declared-push', declared, b'', b'')
        self.assertNotIn('ARR_PUSH', normal)

    def test_mutation_refusals_and_void_results(self):
        bodies = {
            'set-arity': '(array_set values 0)',
            'push-arity': '(array_push values)',
            'length-arity': '(array_length values 1)',
            'receiver': '(array_push 1 2.5)',
            'index': '(array_set values true 2.5)',
            'value': '(array_set values 0 2)',
            'push-value': '(array_push values 2)',
            'unknown-empty': 'let empty = (array_push [] 1.5)',
            'void-array': 'let wrong: array<float> = (array_set values 0 2.5)',
            'void-float': 'let wrong: float = (array_set values 0 2.5)',
            'bound-set': 'let array_set: int = 7 (array_set values 0 2.5)',
            'bound-push': 'let array_push: int = 7 (array_push values 2.5)',
            'bound-length': 'let array_length: int = 7 (array_length values)',
            'array-reassignment': 'let mut alias = values set alias (array_push values 2.5)',
            'owner-field-write': 'set bundle.left values',
        }
        for name, body in bodies.items():
            self.refused('mutation-' + name, self.mutation_fixture(body))
        self.refused('mutation-void-field', self.mutation_fixture('').replace('left: values }', 'left: (array_set values 0 2.5) }'))
        for name, body in {
                'optional-set': '(array_set values 0 (at values 0))',
                'optional-push': '(array_push values (at values 0))'}.items():
            text = self.mutation_fixture(body)
            # I keep public source output atomic and test raw final authority separately.
            self.refused(name, text, shadows=True)
            path = self.work / (name + '.nano')
            path.write_text(text)
            for emitter in self.emitters:
                assembly = self.work / (name + '-' + emitter.name + '.nasm')
                output = assembly.with_suffix('.nvm')
                output.write_bytes(b'previous verified publication')
                self.command(emitter, path, '-o', assembly)
                result = subprocess.run([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', output],
                                        cwd=ROOT, capture_output=True, text=True, timeout=180)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'previous verified publication')
        false_shadow = self.mutation_fixture('(array_push values 2.5)').replace('(main) 0', '(main) 1')
        self.refused('mutation-false-shadow', false_shadow, shadows=True)

    def test_dynamic_set_bounds_and_prepared_failure_cleanup(self):
        base = self.mutation_fixture('(array_set (array_push values 2.5) 99 3.5)')
        prepared_failure = PREFIX + """struct Bundle { file: Handle, values: array<float> }
fn index(owner: Handle) -> int { let Handle { fd } = owner assert (== fd 0) return 0 }
shadow index { assert (== (index Handle { fd: 0 }) 0) }
fn main() -> int {
    let bundle: Bundle = Bundle { file: Handle { fd: 7 }, values: [1.5] }
    let Bundle { file, values } = bundle
    (array_set (array_push values 2.5) (index file) 3.5)
    return 0
}
shadow main { assert (== (main) 0) }
"""
        for name, text in [('high', base), ('negative', base.replace(' 99 ', ' -1 ')),
                           ('prepared-index-failure', prepared_failure)]:
            self.refused('mutation-bounds-' + name, text, shadows=True)
            path = self.work / ('mutation-bounds-' + name + '.nano')
            path.write_text(text)
            for emitter in self.emitters:
                assembly = self.work / ('mutation-bounds-' + name + '-' + emitter.name + '.nasm')
                module = assembly.with_suffix('.nvm')
                self.command(emitter, path, '-o', assembly)
                self.command(ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module)
                self.execute_pair(module, expected=1, expected_output=b'')


def load_tests(loader, standard_tests, pattern):
    mutation = [name for name in SourceOwnedFloatArrayMutation.__dict__ if name.startswith('test_')]
    adjacency = [name for name in source.SourceOwnedFloatArrays.__dict__ if name.startswith('test_')]
    return unittest.TestSuite(SourceOwnedFloatArrayMutation(name) for name in mutation + adjacency)
