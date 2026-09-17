"""I preserve declared types when checking global initializers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GlobalInitializerContext(unittest.TestCase):
    def compile_run(self, declaration, body):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'global.nano'
            output = Path(tmp) / 'native'
            source.write_text(declaration + '\nfn main() -> int {\n' + body +
                              '\nreturn 0\n}\nshadow main { assert true }\n')
            for command in ([ROOT / 'bin/nano_virt', source, '--run'],
                            [ROOT / 'bin/nanoc_c', source, '-o', output], [output]):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_constant_map(self):
        self.compile_run('let values: HashMap<string,int> = (map_new)',
                         '(map_put values "answer" 42)\nassert (== (map_get values "answer") 42)')

    def test_mutable_map(self):
        self.compile_run('let mut values: HashMap<int,string> = (map_new)',
                         '(map_put values 42 "answer")\nassert (== (map_get values 42) "answer")')

    def test_empty_string_array(self):
        self.compile_run('let mut values: array<string> = []',
                         'set values (array_push values "answer")\nassert (== (at values 0) "answer")')

    def test_invalid_global_types_preserve_prior_output(self):
        declarations = ['let values: array<string> = [42]',
                        'let values: HashMap<string,int> = (map_new 1)',
                        'let values: HashMap<bool,int> = (map_new)']
        for declaration in declarations:
            for compiler in ('nanoc_c', 'nano_virt'):
                with self.subTest(declaration=declaration, compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp) / 'bad.nano', Path(tmp) / 'prior'
                    source.write_text(declaration + '\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'previous output')
                    command = [ROOT / 'bin' / compiler, source, '-o', output]
                    if compiler == 'nano_virt': command += ['--emit-nvm']
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous output')


if __name__ == '__main__':
    unittest.main()
