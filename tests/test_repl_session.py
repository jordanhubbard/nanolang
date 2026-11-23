"""I replay accepted source operations and preserve prior output on save failure."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b'# nanolang-repl-session v1\n'


def record(operation, source=b'', origin=None):
    path = os.fsencode(origin or ROOT / '<repl>') if operation != 'clear' else b''
    return f'{operation} {len(path)} {len(source)}\n'.encode() + path + source + b'\n'


class ReplSession(unittest.TestCase):
    def observe(self, lines, cwd=ROOT, extra_env=None):
        binary = Path(os.environ.get('NANO_REPL', ROOT / 'bin/nanolang-repl')).resolve()
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        with tempfile.TemporaryFile() as source:
            source.write(('\n'.join(lines) + '\n').encode())
            source.seek(0)
            result = run_captured([binary], cwd=cwd, stdin=source, env=environment, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertNotIn(b'AddressSanitizer', result.stderr)
        self.assertNotIn(b'runtime error:', result.stderr)
        return result

    def test_loaded_source_survives_removal_of_the_original_file(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source, saved = Path(directory) / 'source.nano', Path(directory) / 'saved.nano'
            source.write_text('fn loaded(x: int) -> int { return (+ x 2) }\n'
                              'shadow loaded { assert (== (loaded 7) 9) }\n')
            result = self.observe([f':load {source}', f':save {saved}', ':quit'])
            self.assertIn(b'saved 1 fragment(s)', result.stdout)
            self.assertTrue(saved.read_bytes().startswith(MAGIC))
            self.assertIn(b'fn loaded', saved.read_bytes())
            source.unlink()
            restored = self.observe([f':load {saved}', '(loaded 7)', ':quit'])
        self.assertIn(b'=> 9', restored.stdout)

    def test_typed_and_loaded_mutations_replay_in_order(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source, saved = Path(directory) / 'source.nano', Path(directory) / 'saved.nano'
            source.write_text('set state (+ state 4)\n')
            self.observe(['let mut state: int = 3', f':load {source}',
                          'set state (+ state 5)', f':save {saved}', ':quit'])
            restored = self.observe([f':load {saved}', 'state', ':quit'])
        self.assertIn(b'=> 12', restored.stdout)

    def test_rejected_source_is_not_exported(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            rejected, saved = Path(directory) / 'rejected.nano', Path(directory) / 'saved.nano'
            rejected.write_text('fn export_rejected_file() -> int { return missing_value }\n')
            self.observe(['let mut state: int = 5',
                          'let export_rejected_type: int = "wrong"',
                          'let export_rejected_syntax: int =', f':load {rejected}',
                          'set state 8', f':save {saved}', ':quit'])
            data = saved.read_bytes()
            self.assertNotIn(b'export_rejected_', data)
            restored = self.observe([f':load {saved}', 'state', ':quit'])
        self.assertIn(b'=> 8', restored.stdout)
        self.assertNotIn(b'missing_value', restored.stderr)

    def test_duplicate_failed_load_does_not_duplicate_the_export(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source, saved = Path(directory) / 'source.nano', Path(directory) / 'saved.nano'
            source.write_text('fn kept() -> int { return 17 }\nshadow kept { assert (== (kept) 17) }\n')
            result = self.observe([f':load {source}', f':load {source}', f':save {saved}', ':quit'])
            self.assertIn(b'already defined', result.stderr)
            self.assertEqual(saved.read_bytes().count(b'fn kept'), 1)
            restored = self.observe([f':load {saved}', '(kept)', ':quit'])
        self.assertIn(b'=> 17', restored.stdout)

    def test_reload_preserves_earlier_results_and_ignores_top_level_effects(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            patch, saved = Path(directory) / 'patch.nano', Path(directory) / 'saved.nano'
            patch.write_text('fn value() -> int { return 2 }\nshadow value { assert (== (value) 2) }\n'
                             'set earlier 99\nprintln "ignored-reload-effect"\n')
            self.observe(['fn value() -> int { return 1 }',
                          'shadow value { assert (== (value) 1) }',
                          'let mut earlier: int = (value)', f':reload {patch}', f':save {saved}', ':quit'])
            patch.write_text('fn unusable(\n')
            restored = self.observe([f':load {saved}', 'earlier', '(value)', ':quit'])
        self.assertIn(b'=> 1', restored.stdout)
        self.assertIn(b'=> 2', restored.stdout)
        self.assertNotIn(b'ignored-reload-effect', restored.stdout)

    def test_failed_reload_is_omitted(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            patch, saved = Path(directory) / 'patch.nano', Path(directory) / 'saved.nano'
            patch.write_text('fn value() -> string { return "rejected-reload-body" }\n')
            self.observe(['fn value() -> int { return 3 }', 'shadow value { assert (== (value) 3) }',
                          f':reload {patch}', f':save {saved}', ':quit'])
            self.assertNotIn(b'rejected-reload-body', saved.read_bytes())
            restored = self.observe([f':load {saved}', '(value)', ':quit'])
        self.assertIn(b'=> 3', restored.stdout)

    def test_clear_replays_variable_retirement_and_preserves_functions(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            saved = Path(directory) / 'saved.nano'
            self.observe(['fn kept() -> int { return 13 }', 'shadow kept { assert (== (kept) 13) }',
                          'let forgotten: string = "retired"', ':clear',
                          'let fresh: int = 19', f':save {saved}', ':quit'])
            restored = self.observe([f':load {saved}', '(kept)', 'fresh', ':env', ':quit'])
        self.assertIn(b'=> 13', restored.stdout)
        self.assertIn(b'=> 19', restored.stdout)
        self.assertNotIn(b'  forgotten\n', restored.stdout)

    def test_origins_preserve_relative_imports_across_directories_and_unicode(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            base = Path(directory); origin, destination = base / 'origin', base / 'elsewhere'
            origin.mkdir(); destination.mkdir()
            child = origin / 'child.nano'
            child.write_text('module Helper\npub fn helper() -> int { return 29 }\n'
                             'shadow helper { assert (== (helper) 29) }\n')
            source, saved = origin / 'é\nroot.nano', destination / 'saved.nano'
            source.write_text('module "child.nano" as dep\nfn loaded() -> int { return (dep.helper) }\n'
                              'shadow loaded { assert (== (loaded) 29) }\n')
            # I load a newline-bearing source through a framed session because
            # terminal meta commands themselves remain line-oriented.
            seed = origin / 'seed.nano'
            seed.write_bytes(MAGIC + record('load', source.read_bytes(), source))
            self.observe([f':load {seed}', f':save {saved}', ':quit'])
            source.unlink(); seed.unlink()
            restored = self.observe([':load saved.nano', '(loaded)', ':quit'], cwd=destination)
        self.assertIn(b'=> 29', restored.stdout)

    def test_symlinked_source_uses_one_captured_origin_for_admission_and_replay(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            base = Path(directory); physical, alias = base / 'physical', base / 'alias'
            physical.mkdir(); alias.mkdir()
            for folder, owner, value in [(physical, 'Physical', 67), (alias, 'Alias', 71)]:
                (folder / 'child.nano').write_text(f'module {owner}\npub fn helper() -> int {{ return {value} }}\n'
                                                  f'shadow helper {{ assert (== (helper) {value}) }}\n')
            source = physical / 'source.nano'
            source.write_text('module "child.nano" as dep\nfn loaded() -> int { return (dep.helper) }\n'
                              'shadow loaded { assert (== (loaded) 67) }\n')
            link, saved = alias / 'entry.nano', base / 'saved.nano'
            link.symlink_to(source)
            admitted = self.observe([f':load {link}', '(loaded)', f':save {saved}', ':quit'])
            self.assertIn(b'=> 67', admitted.stdout)
            self.assertNotIn(b'=> 71', admitted.stdout)
            # I retain the origin even if its old pathname becomes a new symlink.
            source.unlink(); source.symlink_to(alias / 'source.nano')
            (alias / 'source.nano').write_text('fn unrelated() -> int { return 0 }\n')
            restored = self.observe([f':load {saved}', '(loaded)', ':quit'])
        self.assertIn(b'=> 67', restored.stdout)
        self.assertNotIn(b'=> 71', restored.stdout)

    def test_malformed_framing_refuses_before_any_prefix_is_applied(self):
        prefix = record('input', b'let state: int = 99')
        invalid = [b'# nanolang-repl-session v2\n', MAGIC + prefix + b'unknown 0 0\n\n',
                   MAGIC + prefix + b'input 99999999999999999999999999999999999 1\n',
                   MAGIC + prefix + b'load 5 10\nshort', MAGIC + prefix + record('clear', b'not-empty'),
                   MAGIC + prefix + b'input 1 1\n/\0\n']
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source = Path(directory) / 'invalid.nano'
            for data in invalid:
                with self.subTest(data=data[-32:]):
                    source.write_bytes(data)
                    result = self.observe(['let state: int = 5', f':load {source}', 'state', ':quit'])
                    self.assertIn(b'=> 5', result.stdout)
                    self.assertNotIn(b'=> 99', result.stdout)
                    self.assertNotIn(b'I replayed', result.stdout)
                    self.assertTrue(result.stderr)

    def test_capacity_refusal_does_not_apply_clear_records(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source = Path(directory) / 'full.nano'
            source.write_bytes(MAGIC + record('clear') * 4096)
            result = self.observe(['let state: int = 31', f':load {source}', 'state', ':quit'])
        self.assertIn(b'=> 31', result.stdout)
        self.assertIn(b'capacity', result.stderr)

    def test_semantic_replay_failure_reports_its_accepted_prefix(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source, saved = Path(directory) / 'partial.nano', Path(directory) / 'saved.nano'
            source.write_bytes(MAGIC + record('input', b'let state: int = 41') +
                               record('input', b'let rejected_marker: int = "wrong"'))
            result = self.observe([f':load {source}', 'state', f':save {saved}', ':quit'])
            self.assertIn(b'after 1 accepted fragment(s)', result.stderr)
            self.assertNotIn(b'rejected_marker', saved.read_bytes())
            restored = self.observe([f':load {saved}', 'state', ':quit'])
        self.assertIn(b'=> 41', result.stdout)
        self.assertIn(b'=> 41', restored.stdout)

    def test_loading_and_resaving_flattens_the_session(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            first, second = Path(directory) / 'first.nano', Path(directory) / 'second.nano'
            self.observe(['let mut state: int = 43', f':save {first}', ':quit'])
            self.observe([f':load {first}', 'set state 47', f':save {second}', ':quit'])
            self.assertEqual(second.read_bytes().count(MAGIC), 1)
            first.unlink()
            restored = self.observe([f':load {second}', 'state', ':quit'])
        self.assertIn(b'=> 47', restored.stdout)

    def test_empty_first_source_record_reports_zero_accepted_operations(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            source = Path(directory) / 'empty-record.nano'
            source.write_bytes(MAGIC + record('input', b'# no executable source\n'))
            result = self.observe([f':load {source}', ':quit'])
        self.assertIn(b'after 0 accepted fragment(s)', result.stderr)
        self.assertNotIn(b'I replayed', result.stdout)

    def test_publish_failure_preserves_destination_and_removes_temporary(self):
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            target = Path(directory) / 'existing'; target.mkdir()
            marker = target / 'marker'; marker.write_bytes(b'prior content')
            result = self.observe(['let state: int = 53', f':save {target}', ':quit'])
            self.assertEqual(marker.read_bytes(), b'prior content')
            self.assertEqual(list(Path(directory).glob('existing.nano-repl-*')), [])
        self.assertIn(b'I cannot save', result.stderr)
        self.assertNotIn(b'; saved ', result.stdout)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'I use a Linux descriptor fault observer.')
    def test_write_and_close_failures_preserve_prior_file_bytes(self):
        binary = Path(os.environ.get('NANO_REPL', ROOT / 'bin/nanolang-repl')).resolve()
        library = Path(os.environ.get('NANO_REPL_SAVE_FAULTS', ROOT / 'obj/repl_save_faults.so')).resolve()
        self.assertTrue(library.is_file(), library)
        closure = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, timeout=30)
        self.assertEqual(closure.returncode, 0, closure.stderr)
        sanitizers = re.findall(r'(?:libasan[^\s]*|libclang_rt\.asan[^\s]*)\s+=>\s+(\S+)', closure.stdout)
        preload = ':'.join([*sanitizers, str(library)])
        with tempfile.TemporaryDirectory(prefix='nano-repl-session-') as directory:
            target = Path(directory) / 'saved.nano'
            for operation in ['write', 'close']:
                with self.subTest(operation=operation):
                    target.write_bytes(b'prior content')
                    result = self.observe(['let state: int = 59', f':save {target}', ':quit'], extra_env={
                        'LD_PRELOAD': preload, 'NANO_REPL_FAIL_SAVE': operation,
                        'NANO_REPL_FAIL_SAVE_TARGET': str(target),
                    })
                    self.assertEqual(target.read_bytes(), b'prior content')
                    self.assertEqual(list(Path(directory).glob('saved.nano.nano-repl-*')), [])
                    self.assertIn(b'I cannot save', result.stderr)
                    self.assertNotIn(b'; saved ', result.stdout)
            self.observe(['let state: int = 61', f':save {target}', ':quit'])
            self.assertTrue(target.read_bytes().startswith(MAGIC))
            restored = self.observe([f':load {target}', 'state', ':quit'])
            self.assertIn(b'=> 61', restored.stdout)


if __name__ == '__main__':
    unittest.main()
