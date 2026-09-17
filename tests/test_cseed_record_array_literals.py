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
