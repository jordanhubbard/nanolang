"""I exercise timeout observation with real child processes and retained pipes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import ci_stage2_emitter_diagnostic as diagnostic


@unittest.skipUnless(sys.platform == 'linux', 'I sample Linux process groups here.')
class Stage2EmitterDiagnostic(unittest.TestCase):
    def test_native_process_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = diagnostic.controls(root / 'controls')
            self.assertEqual(len(rows), 3)
            self.assertEqual((root / 'controls/complete/stdout').read_bytes(), b'complete\n')
            self.assertEqual((root / 'controls/nonzero/stderr').read_bytes(), b'refused\n')
            sampled = [json.loads(line) for line in
                       (root / 'controls/timeout/processes.jsonl').read_text().splitlines()]
            self.assertTrue(any(p['pid'] == rows[2]['pid']
                                for row in sampled for p in row['processes']))
            self.assertTrue(all(p['group'] == rows[2]['pid']
                                for row in sampled for p in row['processes']))

    def test_observer_retains_timeout_stderr_and_exact_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = 'import signal,sys; print("before-timeout",file=sys.stderr,flush=True); signal.pause()'
            row = diagnostic.observe([sys.executable, '-c', source], root / 'command',
                                     cwd=root, deadline=0.2, interval=0.02)
            self.assertTrue(row['command_timeout'])
            self.assertEqual(row['exit_code'], -9)
            self.assertEqual((root / 'command/stderr').read_bytes(), b'before-timeout\n')
            self.assertEqual(row['stderr_sha256'], diagnostic.digest(root / 'command/stderr'))
            self.assertTrue(row['terminal'] and row['sampler_stopped'])
            self.assertFalse(row['observer_errors'])

    def test_original_command_requires_exact_product_and_private_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, work = base / 'product', base / 'temporary'
            command = [str(root / 'bin/nanoc_stage2'),
                       str(root / 'tests/selfhost_shadow_emitter.nano'), '-o',
                       str(work / 'nano-qualification-original/nano-native-shadows-original/nanoc_stage2')]
            log = f"subprocess.TimeoutExpired: Command '{command!r}' timed out after 300 seconds\n"
            self.assertEqual(diagnostic.original_command(root, log, work), command)
            for old, new in [('300 seconds', '301 seconds'), ('bin/nanoc_stage2', 'bin/nanoc_c'),
                             ('nano-native-shadows-original', 'unowned'),
                             (str(work), str(base / 'outside'))]:
                with self.subTest(replacement=new), self.assertRaises(AssertionError):
                    diagnostic.original_command(root, log.replace(old, new), work)
            Path(command[-1]).parent.mkdir(parents=True)
            with self.assertRaises(AssertionError):
                diagnostic.original_command(root, log, work)

    def test_partial_compiler_files_survive_without_following_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            (work / 'partial.c').write_bytes(b'int partial;\n')
            outside = root / 'outside'
            outside.mkdir()
            (outside / 'unrelated').write_text('outside')
            (work / 'link').symlink_to(outside, target_is_directory=True)
            rows = diagnostic.retain_temporaries(work, root / 'retained')
            self.assertEqual((root / 'retained/tree/partial.c').read_bytes(), b'int partial;\n')
            self.assertFalse((root / 'retained/tree/link').exists())
            self.assertEqual(next(row['link'] for row in rows if row['path'] == 'link'), str(outside))
            with self.assertRaises(AssertionError):
                diagnostic.retain_temporaries(work, work / 'recursive')


if __name__ == '__main__':
    unittest.main()
