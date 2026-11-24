"""I publish a completed native artifact without lending prior output to cc."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_PUBLICATION_COMPILER', ROOT / 'bin/nanoc_c'))
VALID = 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
MISSING = '''extern fn nanolang_publication_missing_symbol() -> int
fn main() -> int {
    unsafe {
        if (== (getenv "NANO_PUBLICATION_LINK_CONTROL") "1") {
            return (nanolang_publication_missing_symbol)
        }
    }
    return 0
}
shadow main { assert (== (main) 0) }
'''
PREVIOUS = b'I retain my previous native output.\n'


class CSeedNativePublication(unittest.TestCase):
    def setUp(self):
        # I retain failed fixtures when my external observer supplies a root.
        parent = os.environ.get('NANOLANG_PUBLICATION_WORK')
        self.work = Path(tempfile.mkdtemp(prefix='native publication ', dir=parent))
        if not parent:
            self.addCleanup(shutil.rmtree, self.work)
        self.source = self.work / 'main.nano'
        self.source.write_text(VALID)
        self.output = self.work / 'program with spaces'
        self.output.write_bytes(PREVIOUS)
        self.output.chmod(0o750)
        self.temporary = self.work / 'temporary-c'
        self.temporary.mkdir()
        self.env = dict(os.environ, NANOLANG_ROOT=str(ROOT), TMPDIR=str(self.temporary))
        self.env.pop('NANO_PUBLICATION_LINK_CONTROL', None)
        self.env.pop('NANO_CFLAGS', None)
        self.env.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)
        self.real_cc = shlex.split(self.env.get('NANO_CC', self.env.get('CC', 'cc')))

    def invoke(self, *options, success=True, env=None):
        args = [str(COMPILER), str(self.source), '-o', str(self.output), *map(str, options)]
        result = subprocess.run(args, cwd=ROOT, env=env or self.env,
                                capture_output=True, text=True, timeout=120)
        (self.work / 'compiler.stdout').write_text(result.stdout)
        (self.work / 'compiler.stderr').write_text(result.stderr)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertFalse([p for p in self.work.glob('.nano-native.*')
                          if not p.name.endswith(('.gcno', '.gcda'))],
                         'I leave no provisional executable')
        return result

    def retained(self):
        self.assertEqual(self.output.read_bytes(), PREVIOUS)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o750)
        self.assertEqual(self.source.read_text(), VALID)

    def run_native(self):
        result = subprocess.run([str(self.output)], env=self.env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.stat().st_mode & 0o111)

    def wrapper(self, body):
        path = self.work / 'cc-control'
        path.write_text('#!/bin/sh\n' + body)
        path.chmod(0o700)
        return dict(self.env, NANO_CC=shlex.quote(str(path)))

    def test_real_linker_refusal_preserves_bytes_and_mode(self):
        self.source.write_text(MISSING)
        result = self.invoke(success=False)
        self.assertIn('nanolang_publication_missing_symbol', result.stderr)
        self.assertEqual(self.output.read_bytes(), PREVIOUS)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o750)
        self.assertEqual(self.source.read_text(), MISSING)

    def test_partial_compiler_output_then_failure_preserves_previous(self):
        env = self.wrapper('while [ "$#" -gt 0 ]; do if [ "$1" = -o ]; then shift; printf partial > "$1"; exit 7; fi; shift; done\nexit 9\n')
        self.invoke(success=False, env=env)
        self.retained()

    def test_success_without_completed_executable_preserves_previous(self):
        for body in ('exit 0\n',
                     'while [ "$#" -gt 0 ]; do if [ "$1" = -o ]; then shift; printf partial > "$1"; exit 0; fi; shift; done\n'):
            with self.subTest(body=body):
                self.invoke(success=False, env=self.wrapper(body))
                self.retained()

    def test_symlink_candidate_is_refused(self):
        env = self.wrapper('while [ "$#" -gt 0 ]; do if [ "$1" = -o ]; then shift; rm -f "$1"; ln -s /bin/true "$1"; exit 0; fi; shift; done\n')
        self.invoke(success=False, env=env)
        self.retained()

    def test_success_replaces_existing_and_creates_new_output(self):
        self.invoke()
        self.run_native()
        self.assertNotEqual(self.output.read_bytes(), PREVIOUS)
        self.output.chmod(0o444)
        self.invoke()
        self.run_native()
        self.output.unlink()
        self.invoke()
        self.run_native()

    def test_success_replaces_output_symlink_without_changing_target(self):
        target = self.work / 'target'
        target.write_bytes(PREVIOUS)
        self.output.unlink()
        self.output.symlink_to(target)
        self.invoke()
        self.assertFalse(self.output.is_symlink())
        self.assertEqual(target.read_bytes(), PREVIOUS)
        self.run_native()

    def test_failed_link_preserves_output_symlink(self):
        target = self.work / 'target'
        target.write_bytes(PREVIOUS)
        self.output.unlink()
        self.output.symlink_to(target)
        self.source.write_text(MISSING)
        self.invoke(success=False)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(target.read_bytes(), PREVIOUS)

    def test_directory_destination_is_retained(self):
        self.output.unlink()
        self.output.mkdir()
        marker = self.output / 'retained'
        marker.write_bytes(PREVIOUS)
        self.invoke(success=False)
        self.assertEqual(marker.read_bytes(), PREVIOUS)

    def test_root_aliases_are_refused_before_writing(self):
        for kind in ('same', 'symlink', 'hardlink'):
            with self.subTest(kind=kind):
                destination = self.work / ('root-' + kind)
                if kind == 'same':
                    destination = self.source
                elif kind == 'symlink':
                    destination.symlink_to(self.source)
                else:
                    os.link(self.source, destination)
                self.output = destination
                self.invoke(success=False)
                self.assertEqual(self.source.read_text(), VALID)
                self.assertEqual(destination.read_text(), VALID)

    def test_import_aliases_are_refused_before_writing(self):
        dependency = self.work / 'dependency.nano'
        text = 'pub fn helper() -> int { return 0 }\nshadow helper { assert (== (helper) 0) }\n'
        dependency.write_text(text)
        self.source.write_text('import "' + str(dependency) + '"\n' + VALID)
        for kind in ('same', 'symlink', 'hardlink'):
            with self.subTest(kind=kind):
                destination = self.work / ('dependency-' + kind)
                if kind == 'same':
                    destination = dependency
                elif kind == 'symlink':
                    destination.symlink_to(dependency)
                else:
                    os.link(dependency, destination)
                self.output = destination
                self.invoke(success=False)
                self.assertEqual(dependency.read_text(), text)

    def test_kept_c_cannot_alias_root_or_previous_output(self):
        kept = Path(str(self.output) + '.c')
        kept.symlink_to(self.source)
        self.invoke('--keep-c', success=False)
        self.retained()
        kept.unlink()
        os.link(self.output, kept)
        self.invoke('--keep-c', success=False)
        self.retained()

    def test_saved_c_cannot_alias_native_output(self):
        self.output = Path(str(self.source) + '.genC')
        self.output.write_bytes(PREVIOUS)
        self.invoke('-S', success=False)
        self.assertEqual(self.output.read_bytes(), PREVIOUS)
        self.assertEqual(self.source.read_text(), VALID)

    def test_diagnostic_root_alias_is_refused(self):
        self.invoke('--llm-diags-json', self.source, success=False)
        self.retained()

    def test_real_c_compilation_refusal_preserves_previous(self):
        header = self.work / 'refused.h'
        header.write_text('#error I retain this real C compiler refusal\n')
        result = self.invoke(success=False, env=dict(self.env, NANO_CFLAGS='-include ' + shlex.quote(str(header))))
        self.assertIn('I retain this real C compiler refusal', result.stderr)
        self.retained()

    def test_temporary_cleanup_failure_prevents_publication(self):
        body = shlex.join(self.real_cc) + ' "$@" || exit $?\n'
        body += 'for argument do case "$argument" in "$TMPDIR"/nanoc_*.c) rm "$argument"; mkdir "$argument"; printf retained > "$argument/retained";; esac; done\n'
        result = self.invoke(success=False, env=self.wrapper(body))
        self.assertIn('I cannot remove temporary C before publication', result.stderr)
        self.retained()
        self.assertEqual(len(list(self.temporary.glob('nanoc_*.c/retained'))), 1)

    def test_real_coverage_notes_and_runtime_data_remain_usable(self):
        self.invoke('--coverage')
        notes = sorted(self.work.glob('.nano-native.*.gcno'))
        self.assertTrue(notes, 'I retain actual compiler coverage notes')
        self.run_native()
        data = sorted(self.work.glob('.nano-native.*.gcda'))
        self.assertTrue(data, 'My published program writes its matching coverage data')
        for path in [*notes, *data]:
            self.assertTrue(path.is_file() and not path.is_symlink())
            self.assertTrue(path.stat().st_size)
            self.assertFalse(path.stat().st_mode & 0o111)

    def test_kept_and_saved_c_remain_available(self):
        self.invoke('--keep-c', '-S')
        self.assertTrue(Path(str(self.output) + '.c').read_text())
        self.assertTrue(Path(str(self.source) + '.genC').read_text())
        self.run_native()


if __name__ == '__main__':
    unittest.main()
