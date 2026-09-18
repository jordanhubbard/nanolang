"""I lower exact scalar expression matches with isolated lexical payloads."""
from pathlib import Path
import subprocess
from tests import test_scalar_union_emission as unions

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'union Choice { Some { number: int }, None {} }\n'


class ScalarMatchValues(unions.ScalarUnionEmission):
    def paired(self, source):
        path = self.work/'values.nano'
        path.write_text(source)
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(compiler=compiler):
                module = self.work/f'value-{compiler}.nvm'
                result = self.command(ROOT/'bin'/compiler, path, '--emit-nvm', '-o', module)
                self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
                self.execute(module)
        prior = None
        for i, emitter in enumerate(self.raw):
            with self.subTest(raw=i):
                assembly, module = self.work/f'value-{i}.nasm', self.work/f'value-{i}.nvm'
                self.command(emitter, path, '-o', assembly)
                text = assembly.read_text()
                if prior is not None:
                    self.assertEqual(text, prior)
                prior = text
                self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
                self.execute(module)
        assembly, module = self.work/'value-shadows.nasm', self.work/'value-shadows.nvm'
        assembly.write_text(self.command(self.shadows, path, '0', 'raw').stdout)
        self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', module)
        self.execute(module)

    def test_all_scalar_results_and_expression_contexts(self):
        self.paired(PREFIX + '''
fn integer(value: Choice) -> int { return match value { Some(p) => p.number None(p) => 0 } }
shadow integer { assert (== (integer Choice.Some { number: 7 }) 7) }
fn boolean(value: Choice) -> bool { return match value { Some(p) => (> p.number 0) None(p) => false } }
shadow boolean { assert (not (boolean Choice.None {})) }
fn floating(value: Choice) -> float { return match value { Some(p) => 1.5 None(p) => -0.0 } }
shadow floating { assert (== (floating Choice.Some { number: 7 }) 1.5) }
fn string_value(value: Choice) -> string { return match value { Some(p) => "selected" None(p) => "empty" } }
shadow string_value { assert (== (string_value Choice.None {}) "empty") }
fn twice(value: int) -> int { return (* value 2) }
shadow twice { assert (== (twice 3) 6) }
fn main() -> int {
 let value: Choice = Choice.Some { number: 7 }
 let empty: Choice = Choice.None {}
 let inferred = match value { Some(p) => p.number None(p) => 0 }
 assert (== inferred 7)
 assert (== (integer value) 7)
 assert (== (integer empty) 0)
 assert (boolean value)
 assert (not (boolean empty))
 assert (== (floating value) 1.5)
 assert (== (floating empty) -0.0)
 assert (== (float_to_string (floating empty)) "-0.0")
 assert (== (string_value value) "selected")
 assert (== (string_value empty) "empty")
 assert (== (twice (match value { Some(p) => p.number None(p) => 0 })) 14)
 return match empty { Some(p) => p.number None(p) => 0 }
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_payload_names_restore_outer_and_sibling(self):
        self.paired(PREFIX + '''
fn check(value: Choice) -> int {
 let payload: int = 3
 let number: int = match value {
  Some(payload) => (+ (match Choice.Some { number: 0 } { Some(payload) => payload.number None(empty) => 0 }) payload.number)
  None(empty) => (- payload 3)
 }
 return (+ number payload)
}
shadow check { assert (== (check Choice.Some { number: 7 }) 10) assert (== (check Choice.None {}) 3) }
fn main() -> int { assert (== (check Choice.Some { number: 7 }) 10) assert (== (check Choice.None {}) 3) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_scrutinee_once_and_only_selected_arm(self):
        self.paired(PREFIX + '''
let mut trace: int = 0
fn mark(value: int) -> int { set trace (+ (* trace 10) value) return value }
shadow mark { set trace 0 assert (== (mark 3) 3) assert (== trace 3) }
fn choose(some: bool) -> Choice {
 if some { let ignored: int = (mark 1) return Choice.Some { number: 7 } }
 let ignored: int = (mark 2)
 return Choice.None {}
}
shadow choose { set trace 0 let value: Choice = (choose true) assert (== trace 1) }
fn main() -> int {
 set trace 0
 let first: int = match (choose true) { Some(p) => (mark 3) None(p) => (mark 4) }
 assert (== first 3) assert (== trace 13)
 set trace 0
 let second: int = match (choose false) { Some(p) => (mark 3) None(p) => (mark 4) }
 assert (== second 4) assert (== trace 24)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_value_match_refusals_preserve_output(self):
        cases = {
            'mixed_types': 'let x: int = match value { Some(p) => p.number None(p) => false }',
            'nested_mismatch': 'let x: int = match value { Some(p) => match value { Some(q) => q.number None(e) => false } None(e) => 0 }',
            'nested_unknown': 'let x: int = match value { Some(p) => match value { Some(q) => q.number Other(e) => 0 } None(e) => 0 }',
            'missing_arm': 'let x: int = match value { Some(p) => p.number }',
            'duplicate_arm': 'let x: int = match value { Some(p) => p.number Some(p) => 0 }',
            'unknown_arm': 'let x: int = match value { Some(p) => p.number Other(p) => 0 }',
            'aggregate_result': 'let x: array<int> = match value { Some(p) => [p.number] None(p) => [0] }',
            'escaped_binding': 'let x: int = match value { Some(p) => p.number None(p) => 0 } let y: int = p.number',
        }
        path, output = self.work/'refuse-value.nano', self.work/'prior-output'
        for name, body in cases.items():
            path.write_text(PREFIX + 'fn main() -> int { let value: Choice = Choice.Some { number: 7 } ' + body + ' return 0 }\nshadow main { assert (== (main) 0) }\n')
            for compiler in [*self.raw, ROOT/'bin/nanoc_stage1', ROOT/'bin/nanoc_stage2']:
                with self.subTest(case=name, compiler=compiler):
                    output.write_bytes(b'prior output')
                    args = [str(compiler), str(path), '-o', str(output)]
                    if compiler.name.startswith('nanoc_stage'):
                        args.append('--emit-nvm')
                    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertRegex(result.stdout+result.stderr, r'(?i)(match|scalar|type|binding|scope)')
                    self.assertNotRegex(result.stdout+result.stderr, r'(?i)(parse error|parsing failed|unexpected token)')
                    self.assertEqual(output.read_bytes(), b'prior output')

    def test_selected_value_shadows_remain_mandatory(self):
        path, output = self.work/'shadow-value.nano', self.work/'prior-shadow'
        for shadow in ('assert false', 'let values: array<int> = match Choice.None {} { Some(p) => [p.number] None(p) => [0] } assert (== (array_length values) 1)'):
            path.write_text(PREFIX + '''fn value() -> int { return match Choice.Some { number: 7 } { Some(p) => p.number None(p) => 0 } }
shadow value { ''' + shadow + ''' }
fn main() -> int { return (- (value) 7) }
shadow main { assert (== (main) 0) }
''')
            for compiler in ('nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(compiler=compiler, shadow=shadow):
                    output.write_bytes(b'prior output')
                    result = subprocess.run([str(ROOT/'bin'/compiler), str(path), '--emit-nvm', '-o', str(output)], cwd=ROOT, text=True, capture_output=True, timeout=180)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertRegex(result.stdout+result.stderr, r'(?i)(shadow|assert|match|scalar)')
                    self.assertNotRegex(result.stdout+result.stderr, r'(?i)(parse error|parsing failed|unexpected token)')
                    self.assertEqual(output.read_bytes(), b'prior output')
