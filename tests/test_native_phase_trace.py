"""I test the phase observer against real native compiler success and refusal."""
import hashlib
import json
import os
import signal
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.native_phase_trace import configure, summarize

ROOT = Path(__file__).resolve().parents[1]


class NativePhaseTrace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='native phase trace ')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        selected = os.environ.get('CC') or 'cc'
        self.environment = configure(self.work, dict(os.environ,
            NANO_CC=selected, NANO_NVM2C=sys.executable))

    def invoke(self, source, extra_inputs=()):
        path = self.work/'source $value.c'
        path.write_text(source)
        product = self.work/'native output'
        arguments = ['-std=c99', '-O1', '-Wall', '-Wextra', '-Werror', str(path), *map(str, extra_inputs), '-o', str(product)]
        command = shlex.split(self.environment['NANO_CC'])+arguments
        result = subprocess.run(command, cwd=ROOT, env=self.environment, capture_output=True, text=True, timeout=30)
        summary = summarize(self.work)
        self.assertFalse(summary['native_lifecycle_admitted'])
        self.assertEqual(summary['phase_count'], 1)
        phase = summary['phases'][0]
        self.assertTrue(phase['completed'])
        self.assertEqual(phase['returncode'], result.returncode)
        self.assertEqual(phase['argv'][-len(arguments):], arguments)
        self.assertTrue(phase['inputs_unchanged'])
        self.assertEqual(phase['tool']['sha256'], phase['tool_after']['sha256'])
        self.assertGreater(phase['tool_elapsed_seconds'], 0)
        return result, product, phase

    def test_real_cc_native_output_and_exact_custody(self):
        support = self.work/'support.c'
        support.write_text('int support(void); int support(void){return 7;}\n')
        runtime = self.work/'support object.o'
        selected = json.loads((self.work/'native-phase-trace/cc.json').read_text())['tool_argv']
        compiled = subprocess.run(selected+['-std=c99', '-c', str(support), '-o', str(runtime)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
        result, product, phase = self.invoke('#include <stdio.h>\nextern int support(void);\n'
            'int main(void){if(support()!=7)return 1;puts("I ran traced C.");return 0;}\n', (runtime,))
        self.assertEqual(len(phase['inputs_before']), 2)
        linked = phase['inputs_before'][1]
        self.assertEqual(linked['sha256'], hashlib.sha256(runtime.read_bytes()).hexdigest())
        self.assertEqual(Path(linked['copy']).read_bytes(), runtime.read_bytes())
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        execution = subprocess.run([product], capture_output=True, text=True, timeout=5)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        self.assertEqual(execution.stdout, 'I ran traced C.\n')
        recorded = phase['outputs'][0]
        self.assertEqual(recorded['sha256'], hashlib.sha256(product.read_bytes()).hexdigest())
        self.assertEqual(Path(recorded['copy']).read_bytes(), product.read_bytes())
        before = phase['inputs_before'][0]
        self.assertEqual(Path(before['copy']).read_bytes(), Path(before['path']).read_bytes())

    def test_real_compiler_refusal_remains_a_failure(self):
        result, product, phase = self.invoke('#error I retain this real compiler refusal\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('I retain this real compiler refusal', result.stderr)
        self.assertFalse(product.exists())
        self.assertEqual(phase['outputs'], [])

    def test_interrupted_real_compiler_keeps_an_incomplete_record(self):
        # I hold this test-owned real preprocessor on stdin, then stop its group.
        command = shlex.split(self.environment['NANO_CC'])+['-E', '-']
        process = subprocess.Popen(command, cwd=ROOT, env=self.environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        try:
            deadline = time.monotonic()+5
            observed = None
            while time.monotonic() < deadline:
                records = list((self.work/'native-phase-trace').glob('phase-*/phase.json'))
                if records:
                    record = json.loads(records[0].read_text())
                    if 'tool_pid' in record:
                        observed = record
                        break
                self.assertIsNone(process.poll(), 'I require a real waiting compiler.')
                time.sleep(0.01)
            self.assertIsNotNone(observed, 'I require a retained actual tool launch.')
            self.assertFalse(observed['completed'])
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=5)
            summary = summarize(self.work)
            self.assertEqual(summary['phase_count'], 1)
            retained = summary['phases'][0]
            self.assertFalse(retained['completed'])
            self.assertEqual(retained['tool_pid'], observed['tool_pid'])
            self.assertNotIn('returncode', retained)
            self.assertNotEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)

    def test_real_marker_recorder_retains_literal_phase_and_order(self):
        recorder = self.environment['NANO_COMPILER_PHASE_RECORDER']
        for phase in ('entry; literal $value', "return ' literal"):
            result = subprocess.run([recorder, phase], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
        markers = summarize(self.work)['driver_markers']
        self.assertEqual([row['phase'] for row in markers],
                         ['entry; literal $value', "return ' literal"])
        self.assertLess(markers[0]['monotonic_seconds'], markers[1]['monotonic_seconds'])
        self.assertTrue(all(row['observer_pid'] > 0 and row['parent_pid'] > 0 for row in markers))

    def test_configuration_preserves_existing_tool_flags(self):
        cc = json.loads((self.work/'native-phase-trace/cc.json').read_text())
        original = shlex.split(os.environ.get('CC') or 'cc')
        self.assertEqual(cc['tool_argv'][1:], original[1:])
        self.assertEqual(len(shlex.split(self.environment['NANO_CC'])), 1)
        translator = json.loads((self.work/'native-phase-trace/nvm2c.json').read_text())
        self.assertEqual(translator['tool_argv'], [sys.executable])
