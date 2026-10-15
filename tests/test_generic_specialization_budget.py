"""I refuse unbounded generic production before replacing an output artifact."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GROWING = '''fn grow(value:T, remaining:int) -> void {
 if (> remaining 0) { (grow [value] (- remaining 1)) }
}
shadow grow { (grow 1 0) }
fn main() -> int { (grow 1 0) return 0 }
shadow main { assert (== (main) 0) }
'''
FINITE = '''fn visit(value:T, remaining:int) -> int {
 if (> remaining 0) { return (visit false (- remaining 1)) }
 return remaining
}
shadow visit { assert (== (visit 13 3) 0) }
fn main() -> int { return (visit "start" 5000) }
shadow main { assert (== (main) 0) }
'''


class GenericSpecializationBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix='nano-generic-budget-')
        override = os.environ.get('NANO_GENERIC_CLI')
        cls.compiler = Path(override) if override else Path(cls.scratch.name) / 'nanoisa_emit'
        if not override:
            result = subprocess.run([str(ROOT / 'bin/nanoc_c'), 'src_nano/nanoisa_emit.nano',
                '-o', str(cls.compiler)], cwd=ROOT, capture_output=True, text=True, timeout=900)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_growth_refusal_preserves_assembly_and_binary_outputs(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-output-') as tmp:
            path, output = Path(tmp) / 'source.nano', Path(tmp) / 'output'
            path.write_text(GROWING)
            for flags in ([], ['--emit-nvm']):
                with self.subTest(flags=flags):
                    output.write_bytes(b'prior-output')
                    result = subprocess.run([str(self.compiler), str(path), *flags, '-o', str(output)],
                        cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('64 recursive instances', result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior-output')
                    self.assertEqual(list(Path(tmp).glob('*.tmp.*')), [])

    def test_finite_polymorphic_recursion_publishes_verified_module(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-finite-') as tmp:
            path, output = Path(tmp) / 'source.nano', Path(tmp) / 'output.nvm'
            path.write_text(FINITE)
            for command in ([self.compiler, path, '--emit-nvm', '-o', output],
                            [ROOT / 'bin/nano_vm', '--verify-only', output],
                            [ROOT / 'bin/nano_vm', output]):
                result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
