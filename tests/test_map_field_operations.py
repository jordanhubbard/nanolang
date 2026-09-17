"""I retain concrete map tags through record projections."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MapFieldOperations(unittest.TestCase):
    def checked(self, command):
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def test_scalar_pairs_and_nested_receivers(self):
        baseline = (ROOT / 'tests/unit/test_map_field_operations.nano').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for key in ('int', 'string'):
                for value in ('int', 'string'):
                    for nested in (False, True):
                        with self.subTest(key=key, value=value, nested=nested):
                            text = baseline.replace('HashMap<string,string>', f'HashMap<{key},{value}>')
                            if key == 'int': text = text.replace('"key"', '7')
                            if value == 'int': text = text.replace('"kept"', '42')
                            if nested:
                                text = 'struct Outer { inner: Holder }\n' + text
                                text = text.replace('let h: Holder = Holder { values: (map_new) }',
                                    'let h: Outer = Outer { inner: Holder { values: (map_new) } }')
                                text = text.replace('h.values', 'h.inner.values')
                            source = work / 'case.nano'
                            source.write_text(text)
                            self.checked([ROOT / 'bin/nanoc_c', source, '-o', work / 'native'])
                            self.checked([work / 'native'])
                            self.checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', work / 'case.nvm'])
                            self.checked([ROOT / 'bin/nano_vm', work / 'case.nvm'])

    def test_imported_record_receiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            module = work / 'model.nano'
            module.write_text('pub struct Holder { values: HashMap<string,string> }\n')
            source = work / 'case.nano'
            baseline = (ROOT / 'tests/unit/test_map_field_operations.nano').read_text()
            body = baseline.split('\n', 1)[1].replace('Holder', 'Model.Holder')
            source.write_text('module "' + str(module) + '" as Model\n' + body)
            self.checked([ROOT / 'bin/nanoc_c', source, '-o', work / 'native'])
            self.checked([work / 'native'])
            self.checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', work / 'case.nvm'])
            self.checked([ROOT / 'bin/nano_vm', work / 'case.nvm'])

    def test_wrong_field_map_types_preserve_prior_output(self):
        cases = ('(map_set h.values 7 "value")', '(map_set h.values "key" 42)',
                 '(map_get h.values)', '(map_set h.values)', '(map_get h.values 7)', '(map_has h.values 7)', '(map_remove h.values 7)',
                 'let wrong: HashMap<string,int> = h.values')
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for body in cases:
                source, output = work / 'bad.nano', work / 'prior'
                source.write_text('struct Holder { values: HashMap<string,string> }\n'
                    'fn main() -> int { let h: Holder = Holder { values: (map_new) } '
                    + body + ' return 0 } shadow main { assert true }\n')
                for compiler in ('nanoc_c', 'nano_virt'):
                    with self.subTest(body=body, compiler=compiler):
                        output.write_text('prior artifact')
                        cmd = [ROOT / 'bin' / compiler, source, '-o', output]
                        if compiler == 'nano_virt': cmd.append('--emit-nvm')
                        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
                        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                        self.assertIn('E001 TYPE MISMATCH', r.stderr)
                        self.assertEqual(output.read_text(), 'prior artifact')


if __name__ == '__main__': unittest.main()
