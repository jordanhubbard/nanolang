"""I execute disabled, literal and refusing phase hooks through real source routes."""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_numeric_builtin_selection as numeric

ROOT = Path(__file__).resolve().parents[1]


class CompilerPhaseMarkerContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command
    exercise = numeric.NumericBuiltinSelection.exercise

    @classmethod
    def setUpClass(cls):
        numeric.NumericBuiltinSelection.setUpClass()

    def source(self, assertion):
        return '''from "src_nano/compiler/native_product.nano" import native_phase_marker
fn main()->int{
 assert '''+assertion+'''
 return 0
}
shadow main{assert (== (main) 0)}
'''

    def test_disabled_recorder_has_no_required_external_tool(self):
        with patch.dict(os.environ, {'NANO_COMPILER_PHASE_RECORDER': ''}):
            self.exercise(self.source('(== (native_phase_marker "disabled") 0)'))

    def test_literal_phase_and_quoted_recorder_path(self):
        output = ROOT/'.test_output/compiler-phase-markers'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="marker ' literal ", dir=output))
        recorder = work/"recorder ' literal"
        command = [sys.executable, str(ROOT/'scripts/native_phase_trace.py'),
                   '--directory', str(work.resolve()), '--mark']
        recorder.write_text('#!/bin/sh\nexec '+' '.join(map(shlex.quote, command))+' "$@"\n')
        recorder.chmod(0o700)
        with patch.dict(os.environ, {'NANO_COMPILER_PHASE_RECORDER': str(recorder.resolve())}):
            self.exercise(self.source('(== (native_phase_marker "literal; $value") 0)'))
        rows = [json.loads(path.read_text()) for path in work.glob('marker-*.json')]
        literal = [row for row in rows if row['phase'] == 'literal; $value']
        self.assertGreaterEqual(len(literal), 2, rows)
        self.assertTrue(all(row['monotonic_seconds'] > 0 for row in literal))

    def test_recorder_refusal_is_observable_without_changing_program_acceptance(self):
        output = ROOT/'.test_output/compiler-phase-markers'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='marker-refusal-', dir=output))
        recorder = work/'refusing-recorder'
        recorder.write_text('#!/bin/sh\nexit 7\n')
        recorder.chmod(0o700)
        with patch.dict(os.environ, {'NANO_COMPILER_PHASE_RECORDER': str(recorder.resolve())}):
            self.exercise(self.source('(!= (native_phase_marker "refusal") 0)'))
