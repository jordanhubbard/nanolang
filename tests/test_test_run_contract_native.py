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
from tests.native_process import native_compile_timeout, run_captured

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/test_resource_effect_checker.nano'
ROUTES = os.environ.get('NANOLANG_RUN_CONTRACT_ROUTES',
    'nanoc_c:native,nano_virt:vm,nanoc_stage1:native,nanoc_stage2:native').split(',')


class NativeTestRunContract(unittest.TestCase):
    def test_original_checker_cli_contracts(self):
        output = ROOT / '.test_output/cli-run-contract-native'
        output.mkdir(parents=True, exist_ok=True)
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
                artifact = work / ('checker.nvm' if route == 'vm' else 'checker')
                command = [str(compiler), str(SOURCE), '-o', str(artifact)]
                if route == 'vm':
                    command.append('--emit-nvm')
                (work/'compile-command.json').write_text(json.dumps(command, indent=2)+'\n')
                budget = native_compile_timeout()
                started = time.monotonic()
                try:
                    result = run_captured(command, cwd=ROOT, timeout=budget)
                except subprocess.TimeoutExpired as error:
                    (work/'compile.stdout').write_bytes(error.stdout or b'')
                    (work/'compile.stderr').write_bytes(error.stderr or b'')
                    (work/'compile-status.json').write_text(json.dumps({'timed_out':True,
                        'partial_output':True,'timeout':error.timeout,
                        'elapsed_seconds':time.monotonic()-started})+'\n')
                    raise
                (work/'compile.stdout').write_bytes(result.stdout)
                (work/'compile.stderr').write_bytes(result.stderr)
                (work/'compile.log').write_bytes(result.stdout+b'\n'+result.stderr)
                (work/'compile-status.json').write_text(json.dumps({'returncode':result.returncode,
                    'timed_out':False,'timeout':budget,
                    'elapsed_seconds':time.monotonic()-started})+'\n')
                self.assertEqual(result.returncode,0,(work/'compile.log').read_text(errors='replace'))
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
