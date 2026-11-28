"""I keep rejected fragments out of live declarations and retain checked names."""
import os
from pathlib import Path
import tempfile
import unittest

from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]


class ReplAdmission(unittest.TestCase):
    def observe(self, lines):
        binary = Path(os.environ.get('NANO_REPL', ROOT / 'bin/nanolang-repl')).resolve()
        with tempfile.TemporaryFile() as source:
            source.write(('\n'.join(lines) + '\n').encode())
            source.seek(0)
            result = run_captured([binary], cwd=ROOT, stdin=source, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertNotIn(b'AddressSanitizer', result.stderr)
        self.assertNotIn(b'runtime error:', result.stderr)
        return result

    def test_rejected_type_can_be_declared_with_a_different_shape(self):
        result = self.observe([
            'struct Phantom { value: int } fn broken() -> int { return missing_value }',
            'struct Phantom { other: int }',
            'fn accept(x: Phantom) -> int { return x.other }',
            'shadow accept { assert (== (accept Phantom { other: 7 }) 7) }',
            '(accept Phantom { other: 7 })', ':quit',
        ])
        self.assertIn(b'=> 7', result.stdout)
        self.assertIn(b'missing_value', result.stderr)
        self.assertNotIn(b'already defined', result.stderr)

    def test_rejected_input_does_not_retain_import_aliases(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-admission-') as directory:
            first, second = Path(directory) / 'first.nano', Path(directory) / 'second.nano'
            first.write_text('module First\npub fn helper() -> int { return 11 }\n'
                             'shadow helper { assert (== (helper) 11) }\n')
            second.write_text('module Second\npub fn helper() -> int { return 22 }\n'
                              'shadow helper { assert (== (helper) 22) }\n')
            result = self.observe([
                f'module "{first}" as chosen fn broken() -> int {{ return missing_value }}',
                f'module "{second}" as chosen', '(chosen.helper)', ':quit',
            ])
        self.assertIn(b'=> 22', result.stdout)
        self.assertNotIn(b'=> 11', result.stdout)

    def test_rejected_loaded_declarations_do_not_block_a_corrected_load(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-admission-') as directory:
            bad, good = Path(directory) / 'bad.nano', Path(directory) / 'good.nano'
            bad.write_text('struct Phantom { value: int }\n'
                           'fn broken() -> int { return missing_value }\n')
            good.write_text('struct Phantom { other: int }\n'
                            'fn accept(x: Phantom) -> int { return x.other }\n'
                            'shadow accept { assert (== (accept Phantom { other: 19 }) 19) }\n')
            result = self.observe([f':load {bad}', f':load {good}',
                                   '(accept Phantom { other: 19 })', ':quit'])
        self.assertIn(b'=> 19', result.stdout)
        self.assertNotIn(b'already defined', result.stderr)

    def test_rejected_context_changes_withdraw_the_first_importers_alias(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-admission-') as directory:
            first, second = Path(directory) / 'first.nano', Path(directory) / 'second.nano'
            first.write_text('module First\npub fn helper() -> int { return 11 }\n'
                             'shadow helper { assert (== (helper) 11) }\n')
            second.write_text('module Second\npub fn helper() -> int { return 22 }\n'
                              'shadow helper { assert (== (helper) 22) }\n')
            result = self.observe([
                f'module FirstOwner module "{first}" as chosen module LastOwner fn broken() -> int {{ return missing_value }}',
                'module FirstOwner', f'module "{second}" as chosen', '(chosen.helper)', ':quit',
            ])
        self.assertIn(b'=> 22', result.stdout)
        self.assertNotIn(b'=> 11', result.stdout)

    def test_failed_module_context_preserves_prior_callables_and_values(self):
        result = self.observe([
            'module RootOwner fn anchor() -> int { return 7 }',
            'shadow anchor { assert (== (anchor) 7) }',
            'let label: string = "retained"',
            'module BadOwner struct Phantom { value: int } fn broken() -> int { return missing_value }',
            'fn after() -> int { return (anchor) }',
            'shadow after { assert (== (after) 7) }', '(after)', 'label', ':quit',
        ])
        self.assertIn(b'=> 7', result.stdout)
        self.assertIn(b'=> retained', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_checked_record_and_record_array_bindings_survive_the_next_input(self):
        result = self.observe([
            'struct Point { value: int }', 'let point: Point = Point { value: 7 }',
            'point.value', 'let points: array<Point> = [point]',
            'fn first(items: array<Point>) -> int { let selected: Point = (array_get items 0) return selected.value }',
            'shadow first { assert (== (first [Point { value: 7 }]) 7) }',
            '(first points)', ':clear',
            'let point: Point = Point { value: 9 }', 'point.value', ':quit',
        ])
        self.assertGreaterEqual(result.stdout.count(b'=> 7'), 2)
        self.assertIn(b'=> 9', result.stdout)
        self.assertNotIn(b'Cannot determine struct type', result.stderr)

    def test_loaded_record_binding_retains_its_nominal_name(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-admission-') as directory:
            path = Path(directory) / 'record.nano'
            path.write_text('struct Point { value: int }\nlet point: Point = Point { value: 23 }\n')
            result = self.observe([f':load {path}', 'point.value', ':quit'])
        self.assertIn(b'=> 23', result.stdout)
        self.assertNotIn(b'Cannot determine struct type', result.stderr)


if __name__ == '__main__':
    unittest.main()
