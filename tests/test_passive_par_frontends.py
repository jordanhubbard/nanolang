"""I retain par identity, scope and conservative eligibility across compilers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PassiveParFrontends(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True,
                              text=True, timeout=120)

    def test_native_stages_retain_bindings_and_scalar_values(self):
        source = ROOT/'tests/nanoisa/fixtures/passive_par.nano'
        with tempfile.TemporaryDirectory(prefix='nano-par-native-') as tmp:
            for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=name):
                    output = Path(tmp)/name
                    built = self.run_command(ROOT/'bin'/name, source, '-o', output)
                    self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                    ran = self.run_command(output)
                    self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
                    self.assertEqual(ran.stdout, '13\n')

    def test_dependencies_mutation_and_effects_refuse_before_publication(self):
        bodies = {
            'earlier sibling': 'let a: int = 1 let b: int = a',
            'later sibling': 'let a: int = b let b: int = 1',
            'self reference': 'let a: int = a',
            'duplicate names': 'let a: int = 1 let a: int = 2',
            'mutable binding': 'let mut a: int = 1',
            'statement': 'assert true',
            'nested par': 'par { let a: int = 1 }',
            'array value': 'let a: array<int> = [1]',
            'effectful call': 'let a: int = (side)',
            'mutable input': 'let a: int = changed',
            'empty': '',
        }
        with tempfile.TemporaryDirectory(prefix='nano-par-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory/'input.nano', directory/'prior'
            for name, body in bodies.items():
                source.write_text('fn side() -> int { (println "effect") return 1 }\n'
                                  'shadow side { assert true }\n'
                                  'fn main() -> int { let mut changed: int = 2 par { '+body+' } return 0 }\n'
                                  'shadow main { assert true }\n')
                for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2', 'nanoisa_emit'):
                    with self.subTest(case=name, compiler=compiler):
                        output.write_text('retained output')
                        result = self.run_command(ROOT/'bin'/compiler, source, '-o', output)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertIn('par', (result.stdout+result.stderr).lower())
                        self.assertEqual(output.read_text(), 'retained output')

    def test_raw_emitter_keeps_unproved_local_input_boundary(self):
        with tempfile.TemporaryDirectory(prefix='nano-par-local-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'prior.nasm'
            source.write_text('fn main() -> int { let outside: int = 1 par { let a: int = outside } return a }\n')
            output.write_text('retained assembly')
            result = self.run_command(ROOT/'bin/nanoisa_emit', source, '-o', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('guarded parameters', result.stderr)
            self.assertEqual(output.read_text(), 'retained assembly')


if __name__ == '__main__':
    unittest.main()
