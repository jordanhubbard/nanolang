"""I retain a timed-out compiler group before cleanup, without extending guests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests import test_file_source_plan as retained_runner


class RetainedProcessEvidence(unittest.TestCase):
    def test_timeout_retains_group_and_unchanged_deadline(self):
        with tempfile.TemporaryDirectory(prefix='nano-retained-timeout-') as directory:
            class Harness:
                work = Path(directory)
                command = classmethod(retained_runner.FileSourcePlan.command.__func__)
            with patch.dict(os.environ, {'CMD_TIMEOUT':'1800'}):
                with self.assertRaises(AssertionError):
                    Harness.command('guest', [sys.executable, '-c',
                        'import time; print("started",flush=True); time.sleep(20)'], timeout=.5)
            status=json.loads((Harness.work/'guest-status.json').read_text())
            command=json.loads((Harness.work/'guest-command.json').read_text())
            self.assertTrue(status['timeout'])
            self.assertTrue(status['group_absent'])
            self.assertEqual(status['cleanup_errors'], [])
            self.assertEqual(command['timeout_seconds'], .5)
            self.assertEqual(status['timeout_seconds'], .5)
            self.assertGreaterEqual(status['elapsed_seconds'], .5)
            self.assertLess(status['elapsed_seconds'], 8)
            self.assertEqual((Harness.work/'guest.stdout').read_text(), 'started\n')
            rows=(Harness.work/'guest-timeout-processes.txt').read_text().splitlines()[1:]
            self.assertTrue(rows)
            self.assertTrue(all(row.split()[2]==str(status['pid']) for row in rows))
            with self.assertRaises(ProcessLookupError):
                os.killpg(status['pid'],0)

    def test_unavailable_diagnostic_does_not_prevent_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='nano-retained-snapshot-refusal-') as directory:
            class Harness:
                work = Path(directory)
                command = classmethod(retained_runner.FileSourcePlan.command.__func__)
            with patch('tests.test_file_source_plan.subprocess.run', side_effect=OSError('no ps')):
                with self.assertRaises(AssertionError):
                    Harness.command('guest',[sys.executable,'-c','import time; time.sleep(20)'],timeout=.1)
            status=json.loads((Harness.work/'guest-status.json').read_text())
            self.assertTrue(status['timeout'] and status['group_absent'])
            self.assertIn('no ps',status['process_snapshot_error'])
            self.assertEqual(status['cleanup_errors'],[])
            with self.assertRaises(ProcessLookupError):
                os.killpg(status['pid'],0)

if __name__=='__main__': unittest.main()
