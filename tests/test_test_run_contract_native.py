"""I execute my unchanged checker CLI with each actual selected compiler route."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from scripts.test_run_contract import execute
from scripts.native_phase_trace import configure as configure_phase_trace, summarize as summarize_phase_trace
from tests.native_process import native_compile_timeout, run_captured

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/test_resource_effect_checker.nano'
ROUTES = os.environ.get('NANOLANG_RUN_CONTRACT_ROUTES',
    'nanoc_c:native,nano_virt:vm,nanoc_stage1:native,nanoc_stage2:native').split(',')


class NativeTestRunContract(unittest.TestCase):
    def test_original_checker_cli_contracts(self):
        output = ROOT / '.test_output/cli-run-contract-native'
        output.mkdir(parents=True, exist_ok=True)
        trace_selection = os.environ.get('NANOLANG_NATIVE_PHASE_TRACE', '0')
        self.assertIn(trace_selection, ('0', '1'))
        for selector in ROUTES:
            provider, route = selector.split(':')
            self.assertIn(provider, ('nanoc_c', 'nano_virt', 'nanoc_stage1', 'nanoc_stage2'))
            self.assertIn(route, ('native', 'vm'))
            with self.subTest(provider=provider, route=route):
                work = Path(tempfile.mkdtemp(prefix=provider+'-', dir=output))
                compiler = ROOT / 'bin' / provider
                before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [compiler, SOURCE, SOURCE.with_suffix('.cases.json'),
                            ROOT/'tests/fixtures/resource-checker/valid.nano',
                            ROOT/'tests/fixtures/resource-checker/unconsumed.nano']}
                (work/'inputs-before.json').write_text(json.dumps(before, indent=2)+'\n')
                artifact = work / ('checker.nvm' if route == 'vm' else 'checker')
                command = [str(compiler), str(SOURCE), '-o', str(artifact)]
                if route == 'vm':
                    command.append('--emit-nvm')
                (work/'compile-command.json').write_text(json.dumps(command, indent=2)+'\n')
                traced = trace_selection == '1' and route == 'native' and provider in ('nanoc_stage1', 'nanoc_stage2')
                build_environment = configure_phase_trace(work, os.environ) if traced else None
                budget = native_compile_timeout()
                started = time.monotonic()
                try:
                    result = run_captured(command, cwd=ROOT, timeout=budget, env=build_environment)
                    compiler_returned = time.monotonic()
                except subprocess.TimeoutExpired as error:
                    (work/'compile.stdout').write_bytes(error.stdout or b'')
                    (work/'compile.stderr').write_bytes(error.stderr or b'')
                    (work/'compile-status.json').write_text(json.dumps({'timed_out':True,
                        'partial_output':True,'timeout':error.timeout,
                        'elapsed_seconds':time.monotonic()-started})+'\n')
                    raise
                finally:
                    if traced:
                        phase_summary = summarize_phase_trace(work)
                (work/'compile.stdout').write_bytes(result.stdout)
                (work/'compile.stderr').write_bytes(result.stderr)
                (work/'compile.log').write_bytes(result.stdout+b'\n'+result.stderr)
                (work/'compile-status.json').write_text(json.dumps({'returncode':result.returncode,
                    'timed_out':False,'timeout':budget,
                    'elapsed_seconds':time.monotonic()-started,
                    'compiler_started_monotonic': started,
                    'compiler_returned_monotonic': compiler_returned})+'\n')
                self.assertEqual(result.returncode,0,(work/'compile.log').read_text(errors='replace'))
                if traced:
                    phases = phase_summary['phases']
                    self.assertTrue(any(row['role'] == 'cc' for row in phases), phase_summary)
                    self.assertTrue(any(row['role'] == 'nvm2c' for row in phases), phase_summary)
                    markers = phase_summary['driver_markers']
                    self.assertEqual([row['phase'] for row in markers], [
                        'inputs-start', 'imports-done', 'lex-done', 'parse-done',
                        'checked-emit-start', 'typecheck-start', 'typecheck-done',
                        'program-lowering-start', 'program-lowering-done',
                        'shadow-lowering-start', 'shadow-lowering-done', 'checked-emit-done',
                        'shadow-assembly-start', 'shadow-assembly-done',
                        'shadow-execution-start', 'shadow-execution-done',
                        'product-assembly-start', 'product-assembly-done',
                        'native-publication-start', 'native-publication-done',
                        'inputs-return', 'driver-return'], phase_summary)
                    self.assertGreaterEqual(markers[0]['monotonic_seconds'], started)
                    self.assertLessEqual(markers[-1]['monotonic_seconds'], compiler_returned)
                    for phase in phases:
                        self.assertTrue(phase['completed'], phase)
                        self.assertEqual(phase['returncode'], 0, phase)
                        self.assertTrue(phase['inputs_unchanged'], phase)
                        self.assertIsNotNone(phase['tool'], phase)
                        self.assertIsNotNone(phase['tool_after'], phase)
                        self.assertEqual(phase['tool']['sha256'], phase['tool_after']['sha256'])
                invocation = [str(ROOT/'bin/nano_vm'),str(artifact),'--'] if route == 'vm' else [str(artifact)]
                status = execute(SOURCE,invocation,60,work/'run.log')
                text=(work/'run.log').read_text(errors='replace')
                self.assertEqual(status,0,text)
                outcomes=[]
                for line in text.splitlines():
                    try:
                        row=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row,dict) and 'accepted' in row:
                        outcomes.append(row)
                self.assertEqual([row['returncode'] for row in outcomes],[2,0,1,2],text)
                self.assertTrue(all(row['accepted'] for row in outcomes),text)
                after={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in before}
                (work/'inputs.json').write_text(json.dumps({'before':before,'after':after},indent=2)+'\n')
                self.assertEqual(before,after)
