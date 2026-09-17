"""I retain par identity, scope and conservative eligibility across compilers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def calculator_scalar_closure():
    """I retain the original helper and loop bodies, including their shadows."""
    source = (ROOT/'examples/language/nl_pi_calculator.nano').read_text()
    definitions = []
    for name in ('int_to_float', 'arctan_series', 'calculate_pi_machin'):
        for prefix in ('fn ', 'shadow '):
            start = source.index(prefix + name + ('(' if prefix == 'fn ' else ' {'))
            opening = source.index('{', start)
            depth, end = 1, opening + 1
            while depth:
                if source[end] == '{': depth += 1
                elif source[end] == '}': depth -= 1
                end += 1
            definitions.append(source[start:end])
    definitions.append('fn main() -> int { (println (calculate_pi_machin 50)) return 0 }')
    definitions.append('shadow main { assert (== (main) 0) }')
    return '\n\n'.join(definitions) + '\n'


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

    def test_existing_callable_par_and_full_calculator_native(self):
        fixtures = [('tests/test_par_blocks.nano', (), ''),
                    ('examples/language/nl_pi_calculator.nano', ('5',), 'Result: 3.14159')]
        with tempfile.TemporaryDirectory(prefix='nano-par-legacy-') as tmp:
            for fixture, arguments, expected in fixtures:
                baseline = None
                for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(fixture=fixture, compiler=compiler):
                        output = Path(tmp)/compiler
                        built = self.run_command(ROOT/'bin'/compiler, ROOT/fixture, '-o', output)
                        self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                        ran = self.run_command(output, *arguments)
                        self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
                        self.assertIn(expected, ran.stdout)
                        if baseline is None: baseline = ran.stdout
                        self.assertEqual(ran.stdout, baseline)

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
            'transitive effect': 'let a: int = (wrapper)',
            'recursive call': 'let a: int = (recursive 0)',
            'global read': 'let a: int = (global_read)',
            'same-name declaration': 'let a: float = (int_to_float 1)',
            'foreign declaration': 'let a: int = (foreign 1)',
            'mutable input': 'let a: int = changed',
            'empty': '',
        }
        with tempfile.TemporaryDirectory(prefix='nano-par-refusal-') as tmp:
            directory = Path(tmp)
            source, output = directory/'input.nano', directory/'prior'
            for name, body in bodies.items():
                source.write_text('fn side() -> int { (println "effect") return 1 }\n'
                                  'shadow side { assert true }\n'
                                  'fn wrapper() -> int { return (side) } shadow wrapper { assert true }\n'
                                  'fn recursive(x: int) -> int { if (== x 0) { return 0 } return (recursive (- x 1)) } shadow recursive { assert true }\n'
                                  'let hidden: int = 7 fn global_read() -> int { return hidden } shadow global_read { assert true }\n'
                                  'fn int_to_float(x: int) -> float { (println x) return 1.0 } shadow int_to_float { assert true }\n'
                                  + ('extern fn foreign(x: int) -> int\n' if name == 'foreign declaration' else '') +
                                  'fn main() -> int { let mut changed: int = 2 par { '+body+' } return 0 }\n'
                                  'shadow main { assert true }\n')
                for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2', 'nanoisa_emit'):
                    with self.subTest(case=name, compiler=compiler):
                        output.write_text('retained output')
                        result = self.run_command(ROOT/'bin'/compiler, source, '-o', output)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        diagnostic = 'extern' if name == 'foreign declaration' and compiler == 'nanoisa_emit' else 'par'
                        self.assertIn(diagnostic, (result.stdout+result.stderr).lower())
                        self.assertEqual(output.read_text(), 'retained output')

    def test_bound_module_calls_keep_distinct_declaration_owners(self):
        with tempfile.TemporaryDirectory(prefix='nano-par-owners-') as tmp:
            directory = Path(tmp)
            for name, value in (('left', 37), ('right', 12)):
                (directory / (name + '.nano')).write_text(
                    'fn base() -> int { return ' + str(value) + ' }\n'
                    'shadow base { assert (== (base) ' + str(value) + ') }\n'
                    'pub fn value() -> int { return (base) }\n'
                    'shadow value { assert (== (value) ' + str(value) + ') }\n')
            source = directory / 'main.nano'
            source.write_text('module "left.nano" as left\nmodule "right.nano" as right\n'
                              'fn main() -> int { par { let a: int = (left.value) '
                              'let b: int = (right.value) } assert (== a 37) assert (== b 12) '
                              '(println (+ a b)) return 0 }\nshadow main { assert (== (main) 0) }\n')
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=compiler):
                    output = directory / compiler
                    built = self.run_command(ROOT/'bin'/compiler, source, '-o', output)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    ran = self.run_command(output)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
                    self.assertEqual(ran.stdout, '49\n')
                    if compiler != 'nanoc_c':
                        module = directory / (compiler + '.nvm')
                        emitted = self.run_command(ROOT/'bin'/compiler, source, '--emit-nvm', '-o', module)
                        self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
                        verified = self.run_command(ROOT/'bin/nano_vm', '--verify-only', module)
                        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
                        dumped = self.run_command(ROOT/'bin/nanoisa', 'dump', module)
                        self.assertEqual(dumped.returncode, 0, dumped.stdout + dumped.stderr)
                        self.assertIn('.passive "', dumped.stdout)
                        ran = self.run_command(ROOT/'bin/nano_vm', module)
                        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
                        self.assertEqual(ran.stdout, '49\n')

    def test_raw_emitter_keeps_unproved_local_input_boundary(self):
        with tempfile.TemporaryDirectory(prefix='nano-par-local-') as tmp:
            source, output = Path(tmp)/'input.nano', Path(tmp)/'prior.nasm'
            source.write_text('fn main() -> int { let outside: int = 1 par { let a: int = outside } return a }\n')
            output.write_text('retained assembly')
            result = self.run_command(ROOT/'bin/nanoisa_emit', source, '-o', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('guarded parameters', result.stdout + result.stderr)
            self.assertEqual(output.read_text(), 'retained assembly')


if __name__ == '__main__':
    unittest.main()
