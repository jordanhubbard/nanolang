"""I test the phase observer against real native compiler success and refusal."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
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

    def invoke(self, source):
        path = self.work/'source $value.c'
        path.write_text(source)
        product = self.work/'native output'
        arguments = ['-std=c99', '-O1', '-Wall', '-Wextra', '-Werror', str(path), '-o', str(product)]
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
        result, product, phase = self.invoke('#include <stdio.h>\nint main(void){puts("I ran traced C.");return 0;}\n')
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

    def test_configuration_preserves_existing_tool_flags(self):
        cc = json.loads((self.work/'native-phase-trace/cc.json').read_text())
        original = shlex.split(os.environ.get('CC') or 'cc')
        self.assertEqual(cc['tool_argv'][1:], original[1:])
        self.assertEqual(len(shlex.split(self.environment['NANO_CC'])), 1)
        translator = json.loads((self.work/'native-phase-trace/nvm2c.json').read_text())
        self.assertEqual(translator['tool_argv'], [sys.executable])
