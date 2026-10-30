"""I distinguish build budgets from guest deadlines and reap descendants."""
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.native_process import native_compile_timeout, native_link_flags, objects_before_archives, run_captured


class NativeProcess(unittest.TestCase):
    def test_archive_order_preserves_inputs_and_links_real_users(self):
        inputs = [Path('first.a'), 'user.o', 'second.a', 'other.o', 'first.a']
        self.assertEqual(objects_before_archives(inputs),
                         ['user.o', 'other.o', Path('first.a'), 'second.a', 'first.a'])
        with tempfile.TemporaryDirectory(prefix='nano-link-order-') as directory:
            root = Path(directory)
            (root / 'provider.c').write_text('int provided(void) { return 41; }\n')
            (root / 'user.c').write_text('int provided(void); int user(void) { return provided()+1; }\n')
            (root / 'main.c').write_text('int user(void); int main(void) { return user()!=42; }\n')
            cc = shlex.split(os.environ.get('CC', 'cc'))
            for name in ('provider', 'user', 'main'):
                result = run_captured([*cc, '-c', root / (name+'.c'), '-o', root / (name+'.o')], timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            archive = root / 'provider.a'
            result = run_captured([*shlex.split(os.environ.get('AR', 'ar')), 'rcs', archive, root / 'provider.o'], timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / 'linked'
            bad = [archive, root / 'user.o', root / 'main.o']
            # Darwin can rescan archives. GNU single-pass links must expose the
            # actual negative case before my corrected order is accepted.
            result = run_captured([*cc, *bad, '-o', output], timeout=30)
            if sys.platform.startswith('linux'):
                linker = run_captured([*cc, '-Wl,--version'], timeout=30)
            else:
                linker = None
            if linker is not None and b'GNU ld' in linker.stdout:
                self.assertNotEqual(result.returncode, 0, 'I require the archive-before-user negative control.')
                self.assertIn(b'provided', result.stderr)
            result = run_captured([*cc, *objects_before_archives(bad), '-o', output], timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(run_captured([output], timeout=5).returncode, 0)

    def test_budget_default_override_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(native_compile_timeout(), 600)
            self.assertEqual(native_compile_timeout(default=900), 900)
        with patch.dict(os.environ, {'CMD_TIMEOUT': '1800'}):
            self.assertEqual(native_compile_timeout(), 1800)
            self.assertEqual(native_compile_timeout(default=900), 1800)
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
