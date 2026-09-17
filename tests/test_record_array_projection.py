"""I preserve nominal array elements through direct and chained projections."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'tests/unit/test_record_array_projection.nano').read_text()

class RecordArrayProjection(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_native_and_vm_accessors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for accessor in ('at', 'array_get'):
                with self.subTest(accessor=accessor):
                    source, binary, module = (root / x for x in ('main.nano', 'main', 'main.nvm'))
                    source.write_text(SOURCE.replace('(at ', '(' + accessor + ' '))
                    self.run_checked([ROOT / 'bin/nanoc_c', source, '-o', binary])
                    self.run_checked([binary])
                    self.run_checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
                    self.run_checked([ROOT / 'bin/nano_vm', module])

    def test_wrong_fields_preserve_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for expression in ('(at table.entries 0).absent', '(at table.entries 0).place.absent',
                               '(array_get (entries) 0).absent'):
                for compiler in ('nanoc_c', 'nano_virt'):
                    with self.subTest(expression=expression, compiler=compiler):
                        source, output = root / 'bad.nano', root / 'prior'
                        source.write_text(SOURCE.replace('(at table.entries 0).name', expression))
                        output.write_text('prior-output')
                        command = [ROOT / 'bin' / compiler, source, '-o', output]
                        if compiler == 'nano_virt': command.append('--emit-nvm')
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('E004 UNKNOWN FIELD', result.stdout + result.stderr)
                        self.assertEqual(output.read_text(), 'prior-output')

if __name__ == '__main__':
    unittest.main()
