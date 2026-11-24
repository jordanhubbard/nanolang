"""I preserve shadow-only product entry, initialization and publication boundaries."""
from pathlib import Path
import os
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeShadowOnlyEntry(unittest.TestCase):
    def invoke(self, directory, source, *options, extra_env=None):
        path = directory / 'input.nano'
        output = directory / 'program'
        path.write_text(source)
        compiler = os.environ.get('NANO_NATIVE_SHADOW_SOURCE_COMPILER') or str(ROOT / 'bin/nanoc_stage2')
        environment = dict(os.environ, TMPDIR=str(directory))
        environment.update(extra_env or {})
        args = [compiler, str(path), '-o', str(output), *options]
        process = subprocess.Popen(args, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr), output

    def test_global_initializer_runs_once_in_shadow_and_product(self):
        source = '''
fn initialize() -> int { (println "initialized") return 7 }
let startup: int = (initialize)
shadow initialize { assert (== startup 7) }
fn read() -> int { return startup }
shadow read { assert (== (read) 7) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            compiled, output = self.invoke(Path(tmp), source)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            self.assertEqual(compiled.stdout, b'')
            self.assertEqual(compiled.stderr.count(b'initialized\n'), 1)
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(executed.stdout, b'initialized\n')

    def test_managed_global_keeps_its_initializer_contract(self):
        source = '''
struct Box { value: int }
let box: Box = Box { value: 7 }
fn read() -> int { return box.value }
shadow read { assert (== (read) 7) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            compiled, output = self.invoke(Path(tmp), source)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_failed_shadow_preserves_prior_product(self):
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            directory = Path(tmp)
            (directory / 'program').write_bytes(b'prior product')
            compiled, output = self.invoke(directory, 'fn answer()->int{return 7} shadow answer {assert false}')
            self.assertNotEqual(compiled.returncode, 0)
            self.assertIn(b'shadow', (compiled.stdout + compiled.stderr).lower())
            self.assertEqual(output.read_bytes(), b'prior product')

    def test_unselected_library_still_requires_entry(self):
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            directory = Path(tmp)
            (directory / 'program').write_bytes(b'prior product')
            compiled, output = self.invoke(directory, 'fn answer()->int{return 7}')
            self.assertNotEqual(compiled.returncode, 0)
            self.assertIn(b'no main', compiled.stdout + compiled.stderr)
            self.assertEqual(output.read_bytes(), b'prior product')

    def test_explicit_main_keeps_its_result(self):
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            compiled, output = self.invoke(Path(tmp), 'fn main()->int{return 7} shadow main {assert (== (main) 7)}')
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 7, executed.stderr)

    def test_source_only_shadow_program_skips_execution(self):
        with tempfile.TemporaryDirectory(prefix='nano-shadow-entry-') as tmp:
            compiled, output = self.invoke(Path(tmp), 'fn answer()->int{return 7} shadow answer {assert false}',
                                           '--target', 'c', extra_env={'NANO_CC': '/bin/false'})
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            self.assertIn(b'without executing shadows', compiled.stdout)
            self.assertNotIn(b'__nanoisa_shadow_', output.read_bytes())


if __name__ == '__main__':
    unittest.main()
