"""I keep module-owned record lists distinct through native helpers."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_RECORD_LIST_COMPILER', ROOT/'bin/nanoc_c'))
OUTPUT_ROOT = os.environ.get('NANO_RECORD_LIST_OUTPUT_ROOT')
RECORD = '''VISIBILITYstruct Item { value: int }
pub fn entries() -> List<Item> {
    let items: List<Item> = (list_Item_new)
    (list_Item_push items Item { value: 17 })
    return items
}
shadow entries {
    let items: List<Item> = (entries)
    assert (== (list_Item_length items) 1)
    (list_Item_free items)
}
pub fn take(items: List<Item>) -> int {
    let item: Item = (list_Item_pop items)
    (list_Item_free items)
    return item.value
}
shadow take { assert (== (take (entries)) 17) }
'''
OTHER = '''pub struct Item { text: string }
pub fn entries() -> List<Item> {
    let items = (list_Item_new)
    (list_Item_push items Item { text: "kept" })
    return items
}
shadow entries {
    let items: List<Item> = (entries)
    assert (== (list_Item_length items) 1)
    (list_Item_free items)
}
pub fn take(items: List<Item>) -> int {
    let item: Item = (list_Item_get items 0)
    (list_Item_free items)
    assert (== item.text "kept")
    return 23
}
shadow take { assert (== (take (entries)) 23) }
'''


class CseedRecordLists(unittest.TestCase):
    def check(self, source, modules, *, reject=False):
        if OUTPUT_ROOT:
            work = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=OUTPUT_ROOT))
        else:
            temporary = tempfile.TemporaryDirectory(prefix='nano-record-list-')
            self.addCleanup(temporary.cleanup)
            work = Path(temporary.name)
        (work/'main.nano').write_text(source)
        for name, text in modules.items():
            (work/name).write_text(text)
        output = work/'program'
        sentinel = b'I retain prior record-list output.\n'
        output.write_bytes(sentinel)
        env = os.environ.copy()
        self.assertNotIn('NANO_SHADOW_TIMEOUT_SECONDS', env)
        env.update(NANO_SHADOW_TRACE='1', NANO_SHADOW_PHASE_TRACE='1')
        args = [str(COMPILER), str(work/'main.nano'), '-o', str(output)]
        compiled = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=120)
        (work/'compiler.stdout').write_bytes(compiled.stdout)
        (work/'compiler.stderr').write_bytes(compiled.stderr)
        report = dict(args=args, exit=compiled.returncode, reject=reject, shadow_timeout='default10',
                      compiler_sha256=hashlib.sha256(COMPILER.read_bytes()).hexdigest(),
                      sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob('*.nano')})
        (work/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        diagnostic = (compiled.stdout+compiled.stderr).decode(errors='replace')
        if reject:
            self.assertNotEqual(compiled.returncode, 0, diagnostic)
            self.assertIn('TYPE MISMATCH', diagnostic)
            self.assertNotIn('C compilation failed', diagnostic)
            self.assertEqual(output.read_bytes(), sentinel)
            return
        self.assertEqual(compiled.returncode, 0, diagnostic)
        self.assertNotEqual(output.read_bytes(), sentinel)
        native = subprocess.run([str(output)], cwd=ROOT, env=env, capture_output=True, timeout=15)
        (work/'native.stdout').write_bytes(native.stdout)
        (work/'native.stderr').write_bytes(native.stderr)
        report.update(native_exit=native.returncode, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        (work/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        self.assertEqual(native.returncode, 0, native.stdout+native.stderr)

    def test_private_module_record_list(self):
        self.check('''module "first.nano" as first
fn main() -> int { assert (== (first.take (first.entries)) 17) return 0 }
shadow main { assert (== (main) 0) }
''', {'first.nano': RECORD.replace('VISIBILITY', '')})

    def test_public_module_record_list(self):
        self.check('''module "first.nano" as first
fn main() -> int { assert (== (first.take (first.entries)) 17) return 0 }
shadow main { assert (== (main) 0) }
''', {'first.nano': RECORD.replace('VISIBILITY', 'pub ')})

    def test_same_spelling_different_shape_owners(self):
        self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int {
    assert (== (first.take (first.entries)) 17)
    assert (== (second.take (second.entries)) 23)
    return 0
}
shadow main { assert (== (main) 0) }
''', {'first.nano': RECORD.replace('VISIBILITY', 'pub '), 'second.nano': OTHER})

    def test_declared_owners_are_independent_of_filenames(self):
        self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int {
    assert (== (first.take (first.entries)) 17)
    assert (== (second.take (second.entries)) 23)
    return 0
}
shadow main { assert (== (main) 0) }
''', {'first.nano': 'module ActualFirst\n'+RECORD.replace('VISIBILITY', 'pub '),
      'second.nano': 'module ActualSecond\n'+OTHER})

    def test_distinct_owner_list_refusal_preserves_output(self):
        self.check('''module "first.nano" as first
module "second.nano" as second
fn main() -> int { return (first.take (second.entries)) }
shadow main { assert (== (main) 0) }
''', {'first.nano': RECORD.replace('VISIBILITY', 'pub '), 'second.nano': OTHER}, reject=True)

    def test_lexical_callable_keeps_source_name(self):
        self.check('''module "first.nano" as first
fn add(left: int, right: int) -> int { return (+ left right) }
shadow add { assert (== (add 2 3) 5) }
fn main() -> int {
    let list_Item_get: fn(int, int) -> int = add
    assert (== (list_Item_get 20 22) 42)
    assert (== (first.take (first.entries)) 17)
    return 0
}
shadow main { assert (== (main) 0) }
''', {'first.nano': RECORD.replace('VISIBILITY', '')})


if __name__ == '__main__':
    unittest.main()
