"""I check persistent REPL declarations, state, refusals and source lifetime."""
import os
from pathlib import Path
import tempfile
import unittest
from tests.native_process import run_captured
ROOT = Path(__file__).resolve().parents[1]


class ReplState(unittest.TestCase):
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

    def test_loaded_forward_calls_see_live_variables(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'forward.nano'
            path.write_text('fn loaded(x: int) -> int { return (+ (helper x) state) }\n'
                            'shadow loaded { assert (== (loaded 1) 8) }\n'
                            'fn helper(x: int) -> int { return (+ x 2) }\n'
                            'shadow helper { assert (== (helper 1) 3) }\n')
            result = self.observe(['let mut state: int = 5', f':load {path}',
                                   '(loaded 7)', 'set state 10', '(loaded 7)', ':quit'])
        self.assertIn(b'=> 14', result.stdout)
        self.assertIn(b'=> 19', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_reload_preserves_variables_and_does_not_replay_effects(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'changed.nano'
            path.write_text('let mut state: int = 999\n'
                            'fn kept(x: int) -> int { return (+ x state) }\n'
                            'shadow kept { assert (== (kept 1) 6) }\n'
                            'set state 999\nprintln "replayed-badly"\n')
            result = self.observe(['let mut state: int = 5',
                                   'fn kept(x: int) -> int { return x }',
                                   'shadow kept { assert (== (kept 1) 1) }',
                                   'let action: fn(int) -> int = kept',
                                   f':reload {path}', '(action 7)', 'state', ':quit'])
        self.assertIn(b'=> 12', result.stdout)
        self.assertIn(b'=> 5', result.stdout)
        self.assertNotIn(b'replayed-badly', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_reload_adds_mutually_recursive_functions(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'recursive.nano'
            path.write_text('fn even(x: int) -> bool { if (== x 0) { return true } return (odd (- x 1)) }\n'
                            'shadow even { assert (even 4) }\n'
                            'fn odd(x: int) -> bool { if (== x 0) { return false } return (even (- x 1)) }\n'
                            'shadow odd { assert (not (odd 4)) }\n')
            result = self.observe([f':reload {path}', '(even 4)', '(odd 4)', ':quit'])
        self.assertIn(b'=> true', result.stdout)
        self.assertIn(b'=> false', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_incompatible_reload_keeps_old_callable_and_refuses_new_body(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'wrong.nano'
            path.write_text('fn kept(x: string) -> string { return x }\n'
                            'shadow kept { assert (== (kept "x") "x") }\n'
                            'fn new_only() -> int { return 99 }\n'
                            'shadow new_only { assert (== (new_only) 99) }\n')
            result = self.observe(['fn kept(x: int) -> int { return x }',
                                   'shadow kept { assert (== (kept 1) 1) }',
                                   'let action: fn(int) -> int = kept',
                                   f':reload {path}', '(action 7)', 'new_only', ':quit'])
        self.assertIn(b'=> 7', result.stdout)
        self.assertIn(b'I retain the existing signature', result.stderr)
        self.assertIn(b'I retain the previous functions', result.stderr)
        self.assertIn(b'new_only', result.stderr)
        self.assertNotIn(b'=> 99', result.stdout)

    def test_invalid_reload_keeps_old_function_and_live_text(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'invalid.nano'
            path.write_text('fn kept(x: int) -> int { return missing_value }\n'
                            'shadow kept { assert (== (kept 1) 1) }\n')
            result = self.observe(['let label: string = "still-live"',
                                   'fn kept(x: int) -> int { return x }',
                                   'shadow kept { assert (== (kept 1) 1) }',
                                   f':reload {path}', '(kept 8)', 'label', ':quit'])
        self.assertIn(b'=> 8', result.stdout)
        self.assertIn(b'still-live', result.stdout)
        self.assertIn(b'I retain the previous functions', result.stderr)

    def test_reload_cannot_replace_extern(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'extern.nano'
            path.write_text('fn foreign_only(x: int) -> int { return x }\n'
                            'shadow foreign_only { assert (== (foreign_only 1) 1) }\n')
            result = self.observe(['extern fn foreign_only(x: int) -> int',
                                   f':reload {path}', ':quit'])
        self.assertIn(b'cannot be redefined', result.stderr)
        self.assertIn(b'I retain the previous functions', result.stderr)

    def test_module_reporting_uses_loaded_environment(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'support.nano'
            path.write_text('pub fn helper(x: int) -> int { return (+ x 3) }\n'
                            'shadow helper { assert (== (helper 1) 4) }\n')
            result = self.observe([f'module "{path}" as support', ':modules',
                                   '(support.helper 4)', ':quit'])
        self.assertIn(b'Modules (', result.stdout)
        self.assertIn(b'support', result.stdout)
        self.assertIn(b'=> 7', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_declared_module_owner_survives_cached_access(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'different_stem.nano'
            path.write_text('module DeclaredContext\n'
                            'pub fn helper(x: int) -> int { return (+ x 4) }\n'
                            'shadow helper { assert (== (helper 1) 5) }\n')
            result = self.observe([f'module "{path}" as first', '(first.helper 3)',
                                   f'module "{path}" as second', '(second.helper 4)',
                                   ':modules', ':quit'])
        self.assertIn(b'=> 7', result.stdout)
        self.assertIn(b'=> 8', result.stdout)
        self.assertIn(b'DeclaredContext', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_nested_module_context_restores_root(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            child = Path(directory) / 'child.nano'
            parent = Path(directory) / 'parent.nano'
            child.write_text('pub fn leaf() -> int { return 17 }\n'
                             'shadow leaf { assert (== (leaf) 17) }\n')
            parent.write_text(f'module "{child}" as nested\n'
                              'pub fn helper() -> int { return (nested.leaf) }\n'
                              'shadow helper { assert (== (helper) 17) }\n')
            result = self.observe([f'module "{parent}" as outer', '(outer.helper)',
                                   'fn root_value() -> int { return 3 }',
                                   'shadow root_value { assert (== (root_value) 3) }',
                                   '(root_value)', ':modules', ':quit'])
        self.assertIn(b'=> 17', result.stdout)
        self.assertIn(b'=> 3', result.stdout)
        self.assertNotIn(b'Undefined', result.stderr)

    def test_failed_module_check_restores_root_context(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'bad_module.nano'
            path.write_text('module FailedContext\n'
                            'pub fn broken() -> int { return missing_value }\n'
                            'shadow broken { assert (== (broken) 1) }\n')
            result = self.observe([f'module "{path}" as refused',
                                   'fn root_value() -> int { return 11 }',
                                   'shadow root_value { assert (== (root_value) 11) }',
                                   '(root_value)', ':quit'])
        self.assertIn(b'Type checking failed', result.stderr)
        self.assertIn(b'=> 11', result.stdout)

    def test_invalid_assignment_is_refused_before_mutating_live_state(self):
        result = self.observe(['let mut state: int = 5',
                               'set state "badly-typed"', 'state', ':quit'])
        self.assertIn(b'=> 5', result.stdout)
        self.assertNotIn(b'badly-typed', result.stdout)
        self.assertIn(b'mismatch', result.stderr.lower())

    def test_global_resource_refusal_does_not_publish_a_value(self):
        result = self.observe(['resource struct Owner { value: int }',
                               'let first: Owner = Owner { value: 7 }',
                               'first', ':quit'])
        self.assertIn(b'global resource ownership is not supported', result.stderr)
        self.assertNotIn(b'=> ', result.stdout)

    def test_unfinished_escape_reports_error_and_keeps_session(self):
        result = self.observe(['"unfinished\\', '(+ 2 3)', ':quit'])
        self.assertIn(b'unterminated', result.stderr.lower())
        self.assertIn(b'=> 5', result.stdout)

    def test_session_capacity_refuses_before_publishing_unowned_ast(self):
        result = self.observe(['1'] * 4097 + [':quit'])
        self.assertEqual(result.stdout.count(b'=> 1'), 4096)
        self.assertIn(b'I cannot retain another REPL input', result.stderr)

    def test_save_clear_and_load_preserve_function_declarations(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-state-') as directory:
            path = Path(directory) / 'saved.nano'
            result = self.observe(['fn kept(x: int) -> int { return x }',
                                   'shadow kept { assert (== (kept 1) 1) }',
                                   'let action: fn(int) -> int = kept',
                                   'let label: string = "saved"', f':save {path}',
                                   ':clear', '(kept 8)', ':env', ':quit'])
            self.assertIn('fn kept', path.read_text())
            restored = self.observe([f':load {path}', '(action 9)', 'label', ':quit'])
        self.assertIn(b'=> 8', result.stdout)
        self.assertIn(b'=> 9', restored.stdout)
        self.assertIn(b'saved', restored.stdout)
        self.assertNotIn(b'Undefined', restored.stderr)


if __name__ == '__main__':
    unittest.main()
