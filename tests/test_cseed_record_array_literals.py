"""I construct native record arrays without C compound-array decay."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/unit/test_cseed_record_array_literals.nano'

class RecordArrayLiterals(unittest.TestCase):
    def run_checked(self, command):
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        return p

    def test_native_and_vm_values_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.run_checked([ROOT/'bin/nanoc_c', SOURCE, '-o', root/'native'])
            native=self.run_checked([root/'native'])
            self.run_checked([ROOT/'bin/nano_virt', SOURCE, '--emit-nvm', '-o', root/'module.nvm'])
            vm=self.run_checked([ROOT/'bin/nano_vm', root/'module.nvm'])
            self.assertEqual(native.stdout, vm.stdout)

    def test_direct_nested_projections_retain_record_identity(self):
        baseline = (ROOT / 'tests/nanoisa/fixtures/record_arrays.nano').read_text()
        baseline = baseline.replace('NSType', 'ValueType')
        direct = baseline.replace('first.name', '(at value.symbols 0).name')
        direct = direct.replace('third.location.line', '(at more 2).location.line')
        direct = direct.replace('    let empty_table:',
                                '    assert (== (at (entries) 1).name \"second\")\n    let empty_table:')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, program in (('typed', baseline), ('at', direct),
                                   ('get', direct.replace('(at ', '(array_get '))):
                with self.subTest(form=label):
                    source = root / (label + '.nano')
                    source.write_text(program)
                    self.run_checked([ROOT/'bin/nanoc_c', source, '-o', root/'native'])
                    self.run_checked([root/'native'])
                    self.run_checked([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', root/'module.nvm'])
                    self.run_checked([ROOT/'bin/nano_vm', root/'module.nvm'])

    def test_record_append_captures_receiver_and_returned_value_in_order(self):
        source_text = """struct Item { number: int, label: string }
struct Holder { items: array<Item> }
let mut events: int = 0
fn receiver(items: array<Item>) -> array<Item> { set events (+ (* events 10) 1) return items }
shadow receiver { let items: array<Item> = [] assert (== (array_length (receiver items)) 0) }
fn item(number: int) -> Item { set events (+ (* events 10) 2) return Item { number: number, label: "kept" } }
shadow item { assert (== (item 7).number 7) }
fn main() -> int {
 let holder: Holder = Holder { items: [Item { number: 0, label: "initial" }] }
 let __nl_arg_0_0: int = 7
 set events 0
 let result: array<Item> = (array_push (receiver holder.items) (item __nl_arg_0_0))
 assert (== events 12)
 assert (== (array_length holder.items) 2)
 assert (== (at result 1).number 7)
 assert (== (at result 1).label "kept")
 let local: array<Item> = []
 let appended: array<Item> = (array_push local (item 9))
 assert (== (at appended 0).number 9)
 return 0
}
shadow main { assert (== (main) 0) }
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'main.nano'
            empty = source_text.replace('[Item { number: 0, label: "initial" }]', '[]')
            empty = empty.replace('(array_length holder.items) 2', '(array_length holder.items) 1')
            empty = empty.replace('(at result 1)', '(at result 0)')
            for initial, program in (('nonempty', source_text), ('empty', empty)):
                with self.subTest(initial=initial):
                    source.write_text(program)
                    self.run_checked([ROOT/'bin/nanoc_c', source, '-o', root/'native'])
                    self.run_checked([root/'native'])
                    self.run_checked([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', root/'module.nvm'])
                    self.run_checked([ROOT/'bin/nano_vm', root/'module.nvm'])

    def test_mixed_nominal_elements_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'bad.nano'; output=root/'prior'
            source.write_text('struct First { a: int } struct Second { a: int }\n'
                'fn main() -> int { let values: array<First> = [First { a: 1 }, Second { a: 2 }] return 0 }\n')
            for compiler in ('nanoc_c','nano_virt'):
                with self.subTest(compiler=compiler):
                    output.write_text('prior-output')
                    command=[ROOT/'bin'/compiler, source, '-o', output]
                    if compiler=='nano_virt': command.append('--emit-nvm')
                    p=subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn('same nominal record type', p.stdout+p.stderr)
                    self.assertEqual(output.read_text(),'prior-output')

if __name__=='__main__': unittest.main()
