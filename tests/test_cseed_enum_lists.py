"""I preserve enum-list identities and native values through accepted calls."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_ENUM_LIST_COMPILER', ROOT/'bin/nanoc_c'))
OUTPUT_ROOT = os.environ.get('NANO_ENUM_LIST_OUTPUT_ROOT')
ENUM = '''enum Mode { Zero, Gap = 17, Low = -3 }
fn entries() -> List<Mode> {
    let items: List<Mode> = (list_Mode_new)
    (list_Mode_push items Mode.Gap)
    return items
}
shadow entries { assert (== (list_Mode_length (entries)) 1) }
fn take(items: List<Mode>) -> Mode {
    let result: Mode = (list_Mode_pop items)
    (list_Mode_free items)
    return result
}
shadow take { assert (== (take (entries)) 17) }
'''
MODULE = '''module OWNER
enum Mode { Value = NUMBER }
pub fn entries() -> List<Mode> {
    let items: List<Mode> = (list_Mode_new)
    (list_Mode_push items Mode.Value)
    return items
}
shadow entries { assert (== (list_Mode_length (entries)) 1) }
pub fn take(items: List<Mode>) -> int {
    let result: Mode = (list_Mode_pop items)
    (list_Mode_free items)
    return result
}
shadow take { assert (== (take (entries)) NUMBER) }
'''


class CseedEnumLists(unittest.TestCase):
    def check(self, source, *, modules=None, reject=False, native_failure=False, diagnostic_contains=None, source_only=False):
        manager = None
        if OUTPUT_ROOT:
            directory = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=OUTPUT_ROOT))
        else:
            manager = tempfile.TemporaryDirectory(prefix='nano-enum-list-')
            directory = Path(manager.name)
            self.addCleanup(manager.cleanup)
        source_path = directory/'main.nano'
        source_path.write_text(source)
        for name, text in (modules or {}).items():
            (directory/name).write_text(text)
        output = directory/('program.c' if source_only else 'program')
        sentinel = b'I retain prior enum-list output.\n'
        output.write_bytes(sentinel)
        env = os.environ.copy()
        env.pop('NANO_ENUM_LIST_EMPTY_POP', None)
        self.assertNotIn('NANO_SHADOW_TIMEOUT_SECONDS', env)
        env['NANO_SHADOW_TRACE'] = '1'
        env['NANO_SHADOW_PHASE_TRACE'] = '1'
        args = [str(COMPILER), *(['--target', 'c'] if source_only else []), str(source_path), '-o', str(output)]
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=120)
        (directory/'compiler.stdout').write_bytes(result.stdout)
        (directory/'compiler.stderr').write_bytes(result.stderr)
        report = dict(args=args, exit=result.returncode, compiler_sha256=hashlib.sha256(COMPILER.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), reject=reject,
                      shadow_timeout='default10', source_only=source_only, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None)
        (directory/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        diagnostic = (result.stdout+result.stderr).decode(errors='replace')
        if reject:
            self.assertNotEqual(result.returncode, 0, diagnostic)
            self.assertNotIn('C compilation failed', diagnostic)
            if diagnostic_contains:
                self.assertIn(diagnostic_contains.lower(), diagnostic.lower())
            else:
                events = []
                for line in result.stderr.splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass
                self.assertTrue(any(event.get('event') == 'shadow-complete' and event['failures'] > 0 for event in events), diagnostic)
            self.assertEqual(output.read_bytes(), sentinel)
            return
        self.assertEqual(result.returncode, 0, diagnostic)
        self.assertNotEqual(output.read_bytes(), sentinel)
        native_output = output
        if source_only:
            self.assertNotIn(b'"event":"shadow-start"', result.stderr)
            native_output = directory/'program'
            compile_args = [os.environ.get('CC', 'cc'), '-std=c11', '-pedantic-errors', '-Wall', '-Wextra', '-Werror',
                            str(output), '-o', str(native_output), '-lm']
            compiled = subprocess.run(compile_args, cwd=ROOT, env=env, capture_output=True, timeout=120)
            (directory/'consumer.stdout').write_bytes(compiled.stdout)
            (directory/'consumer.stderr').write_bytes(compiled.stderr)
            report.update(consumer_args=compile_args, consumer_exit=compiled.returncode)
            (directory/'result.json').write_text(json.dumps(report, indent=2)+'\n')
            self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
        if native_failure:
            env['NANO_ENUM_LIST_EMPTY_POP'] = '1'
        run = subprocess.run([str(native_output)], cwd=ROOT, env=env, capture_output=True, timeout=15)
        (directory/'native.stdout').write_bytes(run.stdout)
        (directory/'native.stderr').write_bytes(run.stderr)
        report.update(native_exit=run.returncode, native_failure=native_failure)
        (directory/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        if native_failure:
            self.assertEqual(run.returncode, 1, run.stdout+run.stderr)
            self.assertIn(b'I cannot pop an empty list.', run.stderr)
        else:
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
        return directory

    def test_existing_enum_values(self):
        self.check((ROOT/'tests/nanoisa/fixtures/enum_values.nano').read_text())

    def test_enum_pressure_and_free(self):
        self.check((ROOT/'tests/nanoisa/fixtures/carrier_enum_pressure.nano').read_text())

    def test_scalar_int64_fixture(self):
        self.check((ROOT/'tests/nanoisa/fixtures/enum_int64_values.nano').read_text())

    def test_source_c_signed64_members(self):
        directory = self.check((ROOT/'tests/nanoisa/fixtures/enum_int64_values.nano').read_text(), source_only=True)
        args = [os.environ.get('CC', 'cc'), '-std=c99', '-pedantic-errors', '-Wall', '-Wextra', '-Werror',
                str(directory/'program.c'), '-o', str(directory/'program-c99'), '-lm']
        compiled = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=120)
        (directory/'c99-compiler.stdout').write_bytes(compiled.stdout)
        (directory/'c99-compiler.stderr').write_bytes(compiled.stderr)
        self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
        run = subprocess.run([str(directory/'program-c99')], cwd=ROOT, capture_output=True, timeout=15)
        (directory/'c99-native.stdout').write_bytes(run.stdout)
        (directory/'c99-native.stderr').write_bytes(run.stderr)
        (directory/'c99.json').write_text(json.dumps(dict(args=args, exit=compiled.returncode,
                                                       native_exit=run.returncode), indent=2)+'\n')
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_source_c_lexical_enum_name_shadow(self):
        self.check('''enum Mode { Value = 17 }
struct Box { Value: int }
fn main() -> int {
    let Mode: Box = Box { Value: 42 }
    assert (== Mode.Value 42)
    return 0
}
shadow main { assert (== (main) 0) }
''', source_only=True)

    def test_source_c_profile_refusals_preserve_output(self):
        self.check('''enum Mode { Value = 17 }
fn main() -> int { let items: array<int> = [17] return 0 }
shadow main { assert (== (main) 0) }
''', source_only=True, reject=True, diagnostic_contains='array value')

    def test_callback_parameter_and_result(self):
        self.check(ENUM+'''fn apply(consumer: fn(List<Mode>) -> Mode) -> Mode { return (consumer (entries)) }
shadow apply { assert (== (apply take) 17) }
fn main() -> int { assert (== (apply take) 17) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_same_named_module_enums(self):
        modules = {name+'.nano': MODULE.replace('OWNER', name).replace('NUMBER', number)
                   for name, number in [('first', '17'), ('second', '-3')]}
        directory = self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int {
    assert (== (first.take (first.entries)) 17)
    assert (== (second.take (second.entries)) -3)
    return 0
}
shadow main { assert (== (main) 0) }
''', modules=modules)
        symbols = subprocess.run(['nm', '-g', str(directory/'program')], capture_output=True, text=True, timeout=10)
        (directory/'global-symbols.stdout').write_text(symbols.stdout)
        (directory/'global-symbols.stderr').write_text(symbols.stderr)
        self.assertEqual(symbols.returncode, 0, symbols.stderr)
        self.assertNotIn('nl_list___nano_', symbols.stdout)

    def test_distinct_module_list_preserves_output(self):
        modules = {name+'.nano': MODULE.replace('OWNER', name).replace('NUMBER', number)
                   for name, number in [('first', '17'), ('second', '-3')]}
        self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int { return (first.take (second.entries)) }
shadow main { assert (== (main) 0) }
''', modules=modules, reject=True, diagnostic_contains='TYPE MISMATCH')

    def test_imported_signed64_enum_lists(self):
        modules = {name+'.nano': MODULE.replace('OWNER', name).replace('NUMBER', number)
                   for name, number in [('first', '9223372036854775807'),
                                        ('second', '-9223372036854775808')]}
        self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int {
    assert (== (first.take (first.entries)) 9223372036854775807)
    assert (== (second.take (second.entries)) -9223372036854775808)
    return 0
}
shadow main { assert (== (main) 0) }
''', modules=modules)

    def test_indirect_distinct_enum_list_preserves_output(self):
        for declaration in ['let items: List<Other> = (list_Other_new)',
                            'let items = (list_Other_new)']:
            with self.subTest(declaration=declaration):
                self.check(ENUM+'''enum Other { Value = 17 }
fn main() -> int {
    let consumer: fn(List<Mode>) -> Mode = take
    DECLARATION
    return (consumer items)
}
shadow main { assert (== (main) 0) }
'''.replace('DECLARATION', declaration), reject=True, diagnostic_contains='TYPE MISMATCH')

    def test_explicit_negative_zero_and_wide_pop(self):
        self.check(ENUM.replace('Low = -3', 'Low = -3, Wide = 4294967297')+'''
fn main() -> int {
    let items: List<Mode> = (entries)
    (list_Mode_push items Mode.Low)
    assert (== (list_Mode_pop items) -3)
    (list_Mode_push items Mode.Zero)
    assert (== (list_Mode_pop items) 0)
    (list_Mode_push items Mode.Wide)
    assert (== (list_Mode_pop items) 4294967297)
    assert (== (list_Mode_get items 0) 17)
    assert (== (take items) 17)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_positive_wide_enum_uses_signed_integer_comparison(self):
        self.check('''enum Positive { Value = 4294967297 }
fn greater(value: Positive) -> bool { return (> value -1) }
shadow greater { assert (greater Positive.Value) }
fn main() -> int {
    assert (greater Positive.Value)
    assert (== (< Positive.Value -1) false)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_signed64_extremes_and_successors(self):
        self.check('''enum Edge { Min = -9223372036854775808, MinNext, Max = 9223372036854775807, Reset = -4294967297, ResetNext }
fn main() -> int {
    assert (== Edge.Min -9223372036854775808)
    assert (== Edge.MinNext -9223372036854775807)
    assert (== Edge.Max 9223372036854775807)
    assert (== Edge.ResetNext -4294967296)
    let items: List<Edge> = (list_Edge_new)
    (list_Edge_push items Edge.Min)
    (list_Edge_push items Edge.Max)
    assert (== (list_Edge_pop items) 9223372036854775807)
    assert (== (list_Edge_pop items) -9223372036854775808)
    (list_Edge_free items)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_enum_overflow_preserves_output(self):
        for variants in ['Value = 9223372036854775808', 'Value = -9223372036854775809',
                         'Value = 9223372036854775807, Overflow']:
            with self.subTest(variants=variants):
                self.check('enum Edge { '+variants+' }\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n',
                           reject=True, diagnostic_contains='signed64 range')

    def test_lexical_callable_keeps_source_name(self):
        self.check('''enum Mode { Value = 17 }
fn add(left: int, right: int) -> int { return (+ left right) }
shadow add { assert (== (add 2 3) 5) }
fn main() -> int {
    let list_Mode_get: fn(int, int) -> int = add
    assert (== (list_Mode_get 20 22) 42)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_record_values_survive_pop_and_free(self):
        self.check('''struct Item { value: int, text: string }
fn read() -> int {
    let items: List<Item> = (list_Item_new)
    (list_Item_push items Item { value: 17, text: "kept" })
    let item: Item = (list_Item_pop items)
    (list_Item_free items)
    assert (== item.text "kept")
    return item.value
}
shadow read { assert (== (read) 17) }
fn main() -> int { assert (== (read) 17) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_wrong_parameter_preserves_output(self):
        self.check(ENUM+'''fn main() -> int { return (take "wrong") }
shadow main { assert (== (main) 0) }
''', reject=True, diagnostic_contains='TYPE MISMATCH')

    def test_unknown_element_preserves_output(self):
        self.check('''fn main() -> int { let items = (list_Absent_new) return 0 }
shadow main { assert (== (main) 0) }
''', reject=True, diagnostic_contains='Unknown list element type')

    def test_failed_enum_shadow_preserves_output(self):
        self.check(ENUM.replace('(take (entries)) 17)', '(take (entries)) 18)')+'''
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', reject=True)

    def test_native_empty_pop_refuses_before_read(self):
        self.check('''enum Mode { Value = 17 }
fn main() -> int {
    unsafe {
        if (== (getenv "NANO_ENUM_LIST_EMPTY_POP") "1") {
            let items: List<Mode> = (list_Mode_new)
            return (list_Mode_pop items)
        }
    }
    return 0
}
shadow main { assert (== (main) 0) }
''', native_failure=True)


if __name__ == '__main__':
    unittest.main()
