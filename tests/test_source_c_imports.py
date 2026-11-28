"""I retain checked imported declarations in self-contained source C."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_SOURCE_C_COMPILER', ROOT / 'bin/nanoc_c'))


class SourceCImports(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get('NANO_SOURCE_C_OUTPUT_ROOT')
        self.work = Path(tempfile.mkdtemp(prefix='source C imports ', dir=parent))
        if not parent:
            self.addCleanup(shutil.rmtree, self.work)
        self.source = self.work / 'main.nano'
        self.output = self.work / 'program.c'
        self.report = self.work / 'shadows.json'
        self.env = dict(os.environ, NANOLANG_ROOT=str(ROOT))
        self.cc = shlex.split(os.environ.get('CC', 'cc'))

    def module(self, name, text):
        path = self.work / (name + '.nano')
        path.write_text(text)
        return path

    def emit(self, source, success=True):
        self.source.write_text(source + '\nshadow main { assert false }\n')
        self.output.write_bytes(b'I retain prior C output.\n')
        result = subprocess.run([str(COMPILER), str(self.source), '--target', 'c', '-o', str(self.output),
                                 '--llm-shadow-json', str(self.report)], cwd=ROOT, env=self.env,
                                capture_output=True, text=True, timeout=60)
        (self.work / 'emit.stdout').write_text(result.stdout)
        (self.work / 'emit.stderr').write_text(result.stderr)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertFalse(self.report.exists(), 'I do not execute source-only shadows')
        self.assertFalse(list(self.work.glob('program.c.tmp.*')))
        if not success:
            self.assertEqual(self.output.read_bytes(), b'I retain prior C output.\n')
        return result

    def native(self, expected=0, extra=None):
        for standard in ('c99', 'c11'):
            binary = self.work / standard
            args = [*self.cc, '-std=' + standard, '-pedantic-errors', '-Wall', '-Wextra', '-Werror',
                    str(self.output), *map(str, extra or []), '-lm', '-o', str(binary)]
            result = subprocess.run(args, cwd=self.work, capture_output=True, text=True, timeout=60)
            (self.work / (standard + '.compile.stderr')).write_text(result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            (self.work / (standard + '.stdout')).write_text(result.stdout)
            (self.work / (standard + '.stderr')).write_text(result.stderr)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_qualified_import_retains_declared_body_without_executing_shadows(self):
        leaf = self.module('leaf', 'pub fn answer() -> int { return 42 }\nshadow answer { assert false }\n')
        self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}')
        self.native(42)

    def test_private_helpers_and_root_functions_keep_distinct_owners(self):
        left = self.module('left', 'fn helper() -> int { return 11 }\nshadow helper { assert false }\npub fn answer() -> int { return (helper) }\nshadow answer { assert false }\n')
        right = self.module('right', 'fn helper() -> int { return 22 }\nshadow helper { assert false }\npub fn answer() -> int { return (helper) }\nshadow answer { assert false }\n')
        self.emit(f'module "{left}" as left\nmodule "{right}" as right\n' + '''fn helper() -> int { return 7 }
shadow helper { assert false }
fn main() -> int { let helper: int = 3 assert (== helper 3) return (+ (left.answer) (right.answer)) }
''')
        self.native(33)

    def test_declared_module_owners_and_root_aliases_do_not_depend_on_filenames(self):
        leaf = self.module('leaf', 'module ActualLeaf\npub fn value() -> int { return 42 }\nshadow value { assert false }\n')
        wrapper = self.module('wrapper', f'module ActualWrapper\nmodule "{leaf}" as dep\npub fn value() -> int {{ return (dep.value) }}\nshadow value {{ assert false }}\n')
        self.emit(f'module ActualRoot\nmodule "{wrapper}" as lib\nfn main() -> int {{ return (lib.value) }}')
        self.native(42)

    def test_two_aliases_share_one_exact_declaration(self):
        leaf = self.module('leaf', 'pub fn answer() -> int { return 21 }\nshadow answer { assert false }\n')
        self.emit(f'module "{leaf}" as first\nmodule "{leaf}" as second\nfn main() -> int {{ return (+ (first.answer) (second.answer)) }}')
        self.native(42)

    def test_transitive_aliases_are_resolved_in_their_declaring_module(self):
        one = self.module('one', 'pub fn value() -> int { return 11 }\nshadow value { assert false }\n')
        two = self.module('two', 'pub fn value() -> int { return 22 }\nshadow value { assert false }\n')
        left = self.module('left', f'module "{one}" as dependency\npub fn value() -> int {{ return (dependency.value) }}\nshadow value {{ assert false }}\n')
        right = self.module('right', f'module "{two}" as dependency\npub fn value() -> int {{ return (dependency.value) }}\nshadow value {{ assert false }}\n')
        self.emit(f'module "{left}" as left\nmodule "{right}" as right\nfn main() -> int {{ return (+ (left.value) (right.value)) }}')
        self.native(33)

    def test_selective_and_plain_imports_keep_checked_function_bindings(self):
        leaf = self.module('leaf', 'pub fn value() -> int { return 42 }\nshadow value { assert false }\n')
        for imported, call in [(f'from "{leaf}" import value as selected', '(selected)'),
                               (f'import "{leaf}"', '(value)')]:
            with self.subTest(imported=imported):
                self.emit(imported + f'\nfn main() -> int {{ return {call} }}')
                self.native(42)

    def test_private_scalar_globals_and_root_shadows_keep_exact_storage(self):
        text = '''let mut state: int = 5
pub fn next() -> int { set state (+ state 1) return state }
shadow next { assert false }
'''
        left = self.module('left', text)
        right = self.module('right', text)
        self.emit(f'module "{left}" as left\nmodule "{right}" as right\n' + '''let mut state: int = 100
fn main() -> int {
 let state: int = 7
 assert (== state 7)
 assert (== (left.next) 6)
 assert (== (left.next) 7)
 assert (== (right.next) 6)
 return 0
}
''')
        self.native()

    def test_public_global_aliases_share_storage_and_initialize_dependencies_first(self):
        leaf = self.module('leaf', '''pub let mut state: int = 5
pub fn next() -> int { set state (+ state 1) return state }
shadow next { assert false }
''')
        self.emit(f'module "{leaf}" as dep\nfrom "{leaf}" import state as shared\n' + '''let observed: int = (dep.next)
fn main() -> int {
 assert (== observed 6)
 assert (== shared 6)
 set shared 9
 assert (== dep.state 9)
 assert (== (dep.next) 10)
 return 0
}
''')
        self.native()

    def test_imported_extern_alias_preserves_the_declared_c_symbol(self):
        leaf = self.module('leaf', 'pub extern fn nanolang_import_probe(value: int) -> int\n')
        extra = self.work / 'foreign.c'
        extra.write_text('#include <stdint.h>\nint64_t nanolang_import_probe(int64_t value) { return value + 1; }\n')
        self.emit(f'from "{leaf}" import nanolang_import_probe as selected\nfn main() -> int {{ unsafe {{ return (selected 41) }} }}')
        self.native(42, [extra])

    def test_imported_main_and_private_namespace_cannot_replace_root_entry(self):
        leaf = self.module('leaf', 'pub fn main() -> int { return 13 }\nshadow main { assert false }\n')
        self.emit(f'module "{leaf}" as lib\n' + """fn nano_cb_0_function_0() -> int { return 7 }
shadow nano_cb_0_function_0 { assert false }
fn main() -> int { return (+ (lib.main) (nano_cb_0_function_0)) }
""")
        self.native(20)

    def test_imported_function_and_global_share_root_spelling_without_rebinding(self):
        leaf = self.module('leaf', 'pub fn value() -> int { return 42 }\nshadow value { assert false }\n')
        self.emit(f'module "{leaf}" as lib\n' + """let value: int = 7
fn main() -> int { assert (== value 7) return (lib.value) }
""")
        self.native(42)

    def test_imported_same_spelled_enum_members_keep_their_declared_values(self):
        one = self.module('one', 'pub enum State { Value = 11 }\npub fn value() -> State { return State.Value }\nshadow value { assert false }\n')
        two = self.module('two', 'pub enum State { Value = 22 }\npub fn value() -> State { return State.Value }\nshadow value { assert false }\n')
        self.emit(f'module "{one}" as one\nmodule "{two}" as two\nfn main() -> int {{ assert (== (one.value) 11) assert (== (two.value) 22) return 0 }}')
        self.native()

    def test_qualified_scalar_arguments_are_evaluated_once_in_source_order(self):
        leaf = self.module('leaf', """let mut count: int = 0
pub fn next() -> int { set count (+ count 1) return count }
shadow next { assert false }
pub fn pair(first: int, second: int) -> int { return (+ (* first 10) second) }
shadow pair { assert false }
""")
        self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.pair (lib.next) (lib.next)) }}')
        self.native(12)

    def test_qualified_record_arguments_use_exact_ordered_value_storage(self):
        leaf = self.module('leaf', """pub struct Point { value: int }
let mut count: int = 0
pub fn next() -> Point { set count (+ count 1) return Point { value: count } }
shadow next { assert false }
pub fn pair(first: Point, second: Point) -> int { return (+ (* first.value 10) second.value) }
shadow pair { assert false }
""")
        self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.pair (lib.next) (lib.next)) }}')
        self.native(12)

    def test_qualified_lifted_argument_exit_stops_later_arguments(self):
        leaf = self.module('leaf', 'pub fn pair(first: int, second: int) -> int { return (+ first second) }\nshadow pair { assert false }\n')
        self.emit(f'module "{leaf}" as lib\n' + """union Choice { Some { number: int }, None {} }
fn main() -> int {
 let chosen: Choice = Choice.Some { number: 1 }
 let value: int = (lib.pair (match chosen { Some(data) => { return 42 } None(_) => { return 42 } })
                          (match chosen { Some(data) => { assert false 7 } None(_) => 7 }))
 return value
}
""")
        self.native(42)

    def test_private_imported_call_is_rejected_before_publication(self):
        leaf = self.module('leaf', 'fn hidden() -> int { return 42 }\nshadow hidden { assert false }\n')
        result = self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.hidden) }}', False)
        self.assertIn('private', result.stderr)

    def test_imported_unsupported_array_profile_remains_a_refusal(self):
        leaf = self.module('leaf', 'pub fn values() -> array<int> { return [1, 2] }\nshadow values { assert false }\n')
        result = self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ return 0 }}', False)
        self.assertIn('C', result.stderr)
        self.assertIn('array', result.stderr.lower())

    def test_imported_nominal_arguments_keep_distinct_declarations(self):
        text = '''pub struct Point { value: int }
pub fn make() -> Point { return Point { value: 9 } }
shadow make { assert false }
pub fn take(value: Point) -> int { return value.value }
shadow take { assert false }
'''
        left = self.module('left', text)
        right = self.module('right', text)
        result = self.emit(f'module "{left}" as left\nmodule "{right}" as right\nfn main() -> int {{ return (left.take (right.make)) }}', False)
        self.assertRegex(result.stderr, 'type|Type|identity|declaration')

    def test_imported_supported_union_retains_exact_result_payload(self):
        leaf = self.module('leaf', """pub union Choice { Some { number: int }, None {} }
pub fn make() -> Choice { return Choice.Some { number: 42 } }
shadow make { assert false }
""")
        self.emit(f'module "{leaf}" as lib\n' + """fn main() -> int {
 let value: lib.Choice = (lib.make)
 match value { Some(data) => { return data.number } None(_) => { return 0 } }
 return 1
}
""")
        self.native(42)

    def test_imported_supported_record_retains_native_fields(self):
        leaf = self.module('leaf', '''pub struct Point { value: int }
pub fn make() -> Point { return Point { value: 42 } }
shadow make { assert false }
''')
        self.emit(f'module "{leaf}" as lib\nfn main() -> int {{ let point: lib.Point = (lib.make) return point.value }}')
        self.native(42)


if __name__ == '__main__':
    unittest.main()
