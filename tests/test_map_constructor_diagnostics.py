"""I reject untyped map constructors before publishing executable output."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MapConstructorDiagnostics(unittest.TestCase):
    def test_invalid_constructors_do_not_publish(self):
        programs = [
            'fn build() -> HashMap<string,int> { return (map_new) }',
            'fn build() -> int { return (map_size (map_new)) }',
            'fn build() -> int { (map_new) return 0 }',
            'fn build() -> int { let m: HashMap<string,int> = (map_new 1) return 0 }',
        ]
        for body in programs:
            for frontend in ('nano_virt', 'nanoc_c'):
                with self.subTest(body=body, frontend=frontend), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    source, output = directory / 'input.nano', directory / 'prior.out'
                    source.write_text(body + '\nshadow build { assert true }\n'
                                      'fn main() -> int { return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'previous artifact')
                    args = [ROOT / 'bin' / frontend, source]
                    if frontend == 'nano_virt':
                        args += ['--emit-nvm']
                    args += ['-o', output]
                    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('map_new', result.stderr)
                    self.assertIn('E001', result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous artifact')

    def test_typed_constructor_still_executes(self):
        source_text = '''fn main() -> int {
 let values: HashMap<string,int> = (map_new)
 (map_put values "answer" 42)
 assert (== (map_get values "answer") 42)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, native = directory / 'input.nano', directory / 'native'
            source.write_text(source_text)
            commands = ([ROOT / 'bin/nano_virt', source, '--run'],
                        [ROOT / 'bin/nanoc_c', source, '-o', native], [native])
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('E001', result.stderr)


if __name__ == '__main__':
    unittest.main()
