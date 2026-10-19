"""I distinguish build budgets from guest deadlines and reap descendants."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.native_process import native_compile_timeout, native_link_flags, run_captured


class NativeProcess(unittest.TestCase):
    def test_budget_default_override_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native_compile_timeout(), 600)
        with patch.dict(os.environ, {'CMD_TIMEOUT': '1800'}):
            self.assertEqual(native_compile_timeout(), 1800)
        for value in ('', '0', '-1', '1.5', 'invalid'):
            with self.subTest(value=value), patch.dict(os.environ, {'CMD_TIMEOUT': value}):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    native_compile_timeout()

    def test_link_flags_inherit_and_preserve_explicit_empty_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native_link_flags(), [])
            os.environ['LDFLAGS'] = '--coverage -L"/path with spaces"'
            self.assertEqual(native_link_flags(), ['--coverage', '-L/path with spaces'])
            os.environ['NANO_LDFLAGS'] = '-fsanitize=address'
            self.assertEqual(native_link_flags(), ['-fsanitize=address'])
            os.environ['NANO_LDFLAGS'] = ''
            self.assertEqual(native_link_flags(), [])

    def test_build_budget_is_explicit_and_guest_deadline_stays_short(self):
        command = [sys.executable, '-c', 'import time; time.sleep(.15); print("done")']
        with patch.dict(os.environ, {'CMD_TIMEOUT': '2'}):
            result = run_captured(command, timeout=.02, build=True, text=True)
            self.assertEqual((result.returncode, result.stdout), (0, 'done\n'))
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                run_captured(command, timeout=.02)
            self.assertEqual(caught.exception.timeout, .02)
        with patch.dict(os.environ, {'CMD_TIMEOUT': 'invalid'}):
            result = run_captured([sys.executable, '-c', 'print(7)'], timeout=5)
            self.assertEqual(result.stdout, b'7\n')

    def test_timeout_kills_descendant_and_reaps_driver(self):
        with tempfile.TemporaryDirectory(prefix='nano-process-cleanup-') as tmp:
            marker = Path(tmp) / 'survived'
            ready = Path(tmp) / 'ready'
            child = ('from pathlib import Path; import time; '
                     f'Path({str(ready)!r}).touch(); time.sleep(3); '
                     f'Path({str(marker)!r}).touch()')
            driver = ('import subprocess,sys,time; from pathlib import Path; '
                      f'p=subprocess.Popen([sys.executable,"-c",{child!r}]); '
                      f'\nwhile not Path({str(ready)!r}).exists(): time.sleep(.005)\n'
                      'print("ready",flush=True); time.sleep(20)')
            actual_popen = subprocess.Popen
            processes = []
            def record(*args, **kwargs):
                process = actual_popen(*args, **kwargs)
                processes.append(process)
                return process
            with patch('tests.native_process.subprocess.Popen', side_effect=record):
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_captured([sys.executable, '-c', driver], timeout=2)
            self.assertTrue(ready.exists(), 'I require a real running descendant.')
            self.assertLess(processes[0].returncode, 0)
            with self.assertRaises(ChildProcessError):
                os.waitpid(processes[0].pid, os.WNOHANG)
            time.sleep(3.1)
            self.assertFalse(marker.exists(), 'I must terminate the compiler descendant.')


if __name__ == '__main__':
    unittest.main()
