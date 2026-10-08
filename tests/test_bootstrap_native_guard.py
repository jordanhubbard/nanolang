"""I check bootstrap's native-host allowance and generated-product refusal."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests.bootstrap_native_guard import retained_input_names

ROOT = Path(__file__).resolve().parents[1]


class BootstrapNativeGuard(unittest.TestCase):
    def test_retained_inputs_require_exact_seed_cache_and_snapshot_names(self):
        with tempfile.TemporaryDirectory(prefix='nano-bootstrap-retained-') as directory:
            work = Path(directory).resolve()
            root = work / 'pinned-host-cache'
            stage = root / '.nano-build-fixture'
            stage.mkdir(parents=True)
            host = work / 'host.c'
            host.write_text('int main(void) { return 0; }\n')
            assembly, obj = stage / '__snapshot_0_0.s', stage / 'host.o'
            subprocess.run(['cc', '-S', host, '-o', assembly], check=True,
                           capture_output=True, timeout=30)
            retained = work / 'retained.json'
            retained.write_text(json.dumps({str(root): retained_input_names(
                'host', {'c_sources': ['first.c', 'second.c'], 'shared_c_sources': []})}))
            marker, calls = work / 'rejected', work / 'calls'
            config = {'compiler': ['cc'], 'native_sources': [],
                      'module_build_roots': [], 'retained_host_inputs': str(retained),
                      'native_marker': str(marker), 'probe_log': str(calls)}
            wrapper = work / 'guard'
            wrapper.write_text('#!' + sys.executable + '\nconfig = ' + repr(config) + '\n' +
                               (ROOT / 'tests/bootstrap_native_guard.py').read_text() + '\nmain(config)\n')
            wrapper.chmod(0o755)
            environment = {**os.environ, 'NANOLANG_BOOTSTRAP_NO_CC': '1'}
            for args in ([assembly, '-c', '-o', obj], [obj, '-o', work / 'host']):
                result = subprocess.run([wrapper, *args], env=environment,
                                        capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(subprocess.run([work / 'host'], timeout=10).returncode, 0)
            self.assertFalse(marker.exists())
            self.assertEqual(calls.read_text(), 'native-host-artifact\nnative-host-artifact\n')
            wrong_root = work / 'unrelated-cache' / stage.name / assembly.name
            wrong_stage = root / 'product' / assembly.name
            generated = stage / 'program.c'
            wrong_index = stage / '__snapshot_0_2.s'
            for source in (wrong_root, wrong_stage, generated, wrong_index):
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(host.read_text() if source.suffix == '.c' else assembly.read_text())
            escaped = stage / '__snapshot_0_0.i'
            escaped.symlink_to(wrong_root)
            for source in (wrong_root, wrong_stage, generated, wrong_index, escaped):
                with self.subTest(source=source):
                    result = subprocess.run([wrapper, source, '-c', '-o', work / 'refused.o'],
                                            env=environment, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 91, result.stderr)
                    self.assertFalse((work / 'refused.o').exists())
            self.assertEqual(len(marker.read_text().splitlines()), 5)

    def test_host_inputs_work_and_generated_product_is_refused(self):
        with tempfile.TemporaryDirectory(prefix='nano-bootstrap-guard-') as directory:
            work = Path(directory).resolve()
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
                direct = subprocess.run(['cc', option], env=environment,
                                        capture_output=True, timeout=30)
                self.assertEqual(identity.returncode, direct.returncode, identity.stderr)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
