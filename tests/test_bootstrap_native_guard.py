"""I check bootstrap's native-host allowance and generated-product refusal."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapNativeGuard(unittest.TestCase):
    def test_host_inputs_work_and_generated_product_is_refused(self):
        with tempfile.TemporaryDirectory(prefix='nano-bootstrap-guard-') as directory:
            work = Path(directory)
            host, product = work / 'declared_host.c', work / 'program.c'
            for source in (host, product):
                source.write_text('int main(void) { return 0; }\n')
            marker, calls = work / 'rejected', work / 'calls'
            config = {'compiler': ['cc'], 'native_sources': [str(host)],
                      'module_build_roots': [], 'native_marker': str(marker),
                      'probe_log': str(calls)}
            wrapper = work / 'guard'
            wrapper.write_text('#!' + sys.executable + '\nconfig = ' + repr(config) + '\n' +
                               (ROOT / 'tests/bootstrap_native_guard.py').read_text() + '\nmain(config)\n')
            wrapper.chmod(0o755)
            environment = {**os.environ, 'NANOLANG_BOOTSTRAP_NO_CC': '1'}
            accepted = subprocess.run([wrapper, host, '-o', work / 'host'], env=environment,
                                      capture_output=True, timeout=30)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(subprocess.run([work / 'host'], timeout=10).returncode, 0)
            rejected = subprocess.run([wrapper, product, '-o', work / 'product'],
                                      env=environment, capture_output=True, timeout=30)
            self.assertEqual(rejected.returncode, 91)
            self.assertTrue(marker.exists())
            self.assertFalse((work / 'product').exists())
            self.assertEqual(calls.read_text(), 'native-host-artifact\n')
            probe = subprocess.run([wrapper, '-E', host], env=environment,
                                   capture_output=True, timeout=30)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertIn('host-cache-probe', calls.read_text())
            marker.unlink()
            for option in ('--version', '-Wl,--version', '-print-prog-name=as'):
                identity = subprocess.run([wrapper, option], env=environment,
                                          capture_output=True, timeout=30)
                self.assertEqual(identity.returncode, 0, identity.stderr)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
