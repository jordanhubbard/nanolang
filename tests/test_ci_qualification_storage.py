"""I exercise real child outputs, failure retention and archive refusal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts import ci_qualification_storage as storage

ROOT = Path(__file__).resolve().parents[1]
shutil_usage = type(storage.shutil.disk_usage(ROOT))


class QualificationStorage(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='nano-storage-controls-')
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.output = self.base / 'archives'

    def child(self, source):
        return [sys.executable, '-c', source]

    def test_success_preserves_bytes_modes_links_and_directory_before_removal(self):
        command = self.child("import os,pathlib,tempfile; p=pathlib.Path(tempfile.gettempdir()); "
                             "(p/'debug.dSYM').mkdir(); (p/'debug.dSYM'/'data').write_bytes(b'abc'*10000); "
                             "(p/'program').write_bytes(b'body'); (p/'program').chmod(0o751); "
                             "os.symlink('program',p/'link'); os.link(p/'program',p/'hardlink')")
        self.assertEqual(storage.run(command, 'native-output-control', self.output, self.base), 0)
        self.assertEqual(list(self.base.glob('nano-qualification-*')), [])
        archive, = self.output.glob('*.tar.gz')
        manifest = json.loads(archive.with_suffix('.gz.json').read_text())
        storage.verify_archive(archive, manifest['entries'])
        with archive.open('rb') as stream:
            self.assertEqual(storage.digest(stream), manifest['archive_sha256'])
        self.assertEqual(manifest['entries']['program']['mode'], 0o751)
        self.assertEqual(manifest['entries']['link']['target'], 'program')
        self.assertEqual(manifest['entries']['debug.dSYM/data']['size'], 30000)
        events = [json.loads(row) for row in next(self.output.glob('*.jsonl')).read_text().splitlines()]
        self.assertEqual([row['event'] for row in events], ['before', 'after', 'archived'])
        self.assertTrue(events[0]['volumes']['temporary']['free'] > 0)
        self.assertEqual(events[1]['volumes']['temporary']['size_probe_returncode'], 0)

    def test_failed_real_child_retains_originals_and_exit_status(self):
        status = storage.run(self.child("import pathlib,tempfile,sys; "
                             "(pathlib.Path(tempfile.gettempdir())/'failure.log').write_text('first failure'); sys.exit(7)"),
                             'failed-control', self.output, self.base)
        self.assertEqual(status, 7)
        root, = self.base.glob('nano-qualification-*')
        self.assertEqual((root/'failure.log').read_text(), 'first failure')
        self.assertEqual(list(self.output.glob('*.tar.gz')), [])
        events = [json.loads(row)['event'] for row in next(self.output.glob('*.jsonl')).read_text().splitlines()]
        self.assertEqual(events[-1], 'retained_failure')

    def test_low_space_refuses_to_launch_and_retains_the_fresh_root(self):
        command = self.child("raise AssertionError('I must not start')")
        with mock.patch.object(storage.shutil, 'disk_usage', return_value=shutil_usage(100, 99, 1)):
            with self.assertRaisesRegex(RuntimeError, 'log reserve'):
                storage.run(command, 'low-space', self.output, self.base)
        root, = self.base.glob('nano-qualification-*')
        self.assertTrue(root.is_dir())
        self.assertFalse(list(self.output.glob('*.tar.gz')))

    def test_late_archive_error_never_signals_an_exited_process_group(self):
        signal_group = os.killpg
        with mock.patch.object(storage.os, 'killpg', wraps=signal_group) as signals:
            with mock.patch.object(storage, 'archive_success', side_effect=ValueError('archive refusal')):
                with self.assertRaisesRegex(ValueError, 'archive refusal'):
                    storage.run(self.child('pass'), 'late-error', self.output, self.base)
        self.assertTrue(signals.call_args_list)
        self.assertTrue(all(call.args[1] == 0 for call in signals.call_args_list))
        root, = self.base.glob('nano-qualification-*')
        self.assertTrue(root.is_dir())

    def test_archive_verification_failure_never_removes_source(self):
        root = self.base/'owned';root.mkdir();(root/'input').write_text('unchanged')
        with mock.patch.object(storage, 'verify_archive', side_effect=ValueError('damaged archive')):
            with self.assertRaisesRegex(ValueError, 'damaged archive'):
                storage.archive_success(root, self.base/'evidence.tar.gz')
        self.assertEqual((root/'input').read_text(), 'unchanged')
        self.assertTrue((self.base/'evidence.tar.gz.partial').is_file())

    def test_source_mutation_during_archive_retains_both_versions(self):
        root = self.base/'owned';root.mkdir();(root/'input').write_text('before')
        verify = storage.verify_archive
        def change_after_verifying(*args):
            verify(*args)
            (root/'input').write_text('after')
        with mock.patch.object(storage, 'verify_archive', side_effect=change_after_verifying):
            with self.assertRaisesRegex(ValueError, 'changed during archival'):
                storage.archive_success(root, self.base/'evidence.tar.gz')
        self.assertEqual((root/'input').read_text(), 'after')
        with tarfile.open(self.base/'evidence.tar.gz.partial') as archive:
            self.assertEqual(archive.extractfile('input').read(), b'before')

    def test_nested_phase_archive_outlives_outer_temporary_root(self):
        script = str(ROOT/'scripts/ci_qualification_storage.py')
        child = self.child("import pathlib,tempfile; (pathlib.Path(tempfile.gettempdir())/'nested').write_text('evidence')")
        command = [sys.executable, script, 'run', '--label', 'inner', '--', *child]
        self.assertEqual(storage.run(command, 'outer', self.output, self.base), 0)
        self.assertEqual(len(list(self.output.glob('*.tar.gz'))), 2)
        self.assertEqual(list(self.base.glob('nano-qualification-*')), [])

    def test_cli_runs_every_original_module_in_order_even_after_failure(self):
        with mock.patch.object(storage, 'run', side_effect=[3, 0, 4]) as runner:
            self.assertEqual(storage.main(['unittest', '--', '-v', 'tests.first', 'tests.second', 'tests.third']), 3)
        self.assertEqual([call.args[0][-1] for call in runner.call_args_list],
                         ['tests.first', 'tests.second', 'tests.third'])
        self.assertTrue(all(call.args[0][1:4] == ['-m', 'unittest', '-v'] for call in runner.call_args_list))

    def test_cli_unknown_collection_options_refuse_before_running(self):
        with mock.patch.object(storage, 'run') as runner:
            with self.assertRaises(SystemExit):
                storage.main(['unittest', '--', '-v', 'discover'])
        runner.assert_not_called()

    def test_make_hook_keeps_the_exact_original_module_selection(self):
        original = json.loads((ROOT/'tests/fixtures/qualification-reconstruction-commands.json').read_text())['commands']
        changed = (ROOT/'Makefile.gnu').read_text()
        def commands(source):
            return [line.strip().replace('$(RECONSTRUCTION_UNITTEST)', 'python3 -m unittest')
                    for line in source.splitlines() if line.startswith('\t') and
                    ('tests.test_owned_reconstruction ' in line or 'tests.test_reconstruction_' in line or
                     'tests.test_scalar_reconstruction' in line)]
        self.assertEqual(commands(changed), original)


if __name__ == '__main__':
    unittest.main()
