"""I check real process outcomes and original comprehensive-runner publication boundaries."""
import json
import os
from pathlib import Path
import shutil
import signal
import time
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'scripts/test_run_contract.py'


class TestRunContract(unittest.TestCase):
    def execute(self, directory, program, contract=None, timeout=5):
        source = directory / 'case.nano'
        source.write_text('fn main()->int{return 0}\nshadow main{assert (== (main) 0)}\n')
        if contract is not None:
            source.with_suffix('.cases.json').write_text(json.dumps(contract))
        log = directory / 'run.log'
        result = subprocess.run([sys.executable, ADAPTER, '--source', source, '--log', log,
                                 '--timeout', str(timeout), '--', sys.executable, '-c', program],
                                capture_output=True, text=True, timeout=10)
        return result, log.read_text()

    def document(self, cases):
        return {'schema': 'nanolang-test-run@1', 'cases': cases}

    def test_default_success_failure_and_signal(self):
        for program, expected in [('print("checked")', 0), ('raise SystemExit(3)', 1),
                                  ('import os,signal; os.kill(os.getpid(),signal.SIGTERM)', 1)]:
            with self.subTest(program=program), tempfile.TemporaryDirectory() as tmp:
                result, log = self.execute(Path(tmp), program)
                self.assertEqual(result.returncode, expected, log)

    def test_explicit_positive_refusal_and_arity_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'input.nano').write_text('ownership input')
            contract = self.document([
                {'name': 'arity', 'args': [], 'exit_code': 2},
                {'name': 'positive', 'args': ['accepted'], 'exit_code': 0},
                {'name': 'refusal', 'args': [{'input':'input.nano'}], 'exit_code':1, 'contains':['ownership input']},
            ])
            program = 'import pathlib,sys\nif len(sys.argv)==1:sys.exit(2)\nif sys.argv[1]=="accepted":sys.exit(0)\nprint(pathlib.Path(sys.argv[1]).read_text());sys.exit(1)'
            result, log = self.execute(root, program, contract)
            self.assertEqual(result.returncode, 0, log)
            self.assertEqual(log.count('"accepted": true'), 3)

    def test_refusal_diagnostic_cannot_reuse_an_earlier_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = self.document([
                {'name':'first','args':['first'],'exit_code':1,'contains':['ownership']},
                {'name':'second','args':['second'],'exit_code':1,'contains':['ownership']},
            ])
            result, log = self.execute(Path(tmp), 'import sys\nprint("ownership" if sys.argv[1]=="first" else "different refusal");sys.exit(1)', contract)
            self.assertEqual(result.returncode, 1, log)
            self.assertIn('"accepted": false', log)

    def test_invalid_metadata_and_missing_inputs_execute_nothing(self):
        invalid = [self.document([]), self.document([{'name':'bad','args':[],'exit_code':-1}]),
                   self.document([{'name':'bad','args':[{'input':'missing.nano'}],'exit_code':0}]),
                   self.document([{'name':'bad','args':[{'input':'../outside.nano'}],'exit_code':0}]),
                   self.document([{'name':'bad','args':[],'exit_code':0,'unknown':True}])]
        for contract in invalid:
            with self.subTest(contract=contract), tempfile.TemporaryDirectory() as tmp:
                marker = Path(tmp)/'executed'
                result, log = self.execute(Path(tmp), f'from pathlib import Path; Path({str(marker)!r}).touch()', contract)
                self.assertEqual(result.returncode, 1, log)
                self.assertFalse(marker.exists())

    def test_total_deadline_stops_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'later'
            contract = self.document([
                {'name':'slow','args':['slow'],'exit_code':0},
                {'name':'later','args':['later'],'exit_code':0},
            ])
            program = f'import pathlib,sys,time\nif sys.argv[1]=="slow":time.sleep(5)\nelse:pathlib.Path({str(marker)!r}).touch()'
            result, log = self.execute(Path(tmp), program, contract, timeout=.1)
            self.assertEqual(result.returncode, 1, log)
            self.assertFalse(marker.exists())
            self.assertEqual(log.count('"timed_out": true'), 2)

    def test_interrupt_reaps_the_active_test_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'case.nano';source.write_text('fn main()->int{return 0}')
            marker=root/'pid';log=root/'run.log'
            program=f'import os,pathlib,time;p=pathlib.Path({str(marker)!r}+".tmp");p.write_text(str(os.getpid()));p.replace({str(marker)!r});time.sleep(10)'
            process=subprocess.Popen([sys.executable,ADAPTER,'--source',source,'--log',log,
                '--timeout','15','--',sys.executable,'-c',program],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+5
                while not marker.exists() and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue(marker.exists())
                child=int(marker.read_text())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5),1)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child,0)
                self.assertIn('interrupted',log.read_text())
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)

    def runner(self, mode, prior=False, contract=False):
        with tempfile.TemporaryDirectory(prefix='nano-comprehensive-contract-') as tmp:
            root = Path(tmp)
            (root/'tests/unit').mkdir(parents=True)
            (root/'scripts').mkdir()
            (root/'bin').mkdir()
            shutil.copy(ROOT/'tests/run_all_tests.sh', root/'tests/run_all_tests.sh')
            shutil.copy(ADAPTER, root/'scripts/test_run_contract.py')
            source = root/'tests/nl_syntax_fixture.nano'
            source.write_text('fn main()->int{return 0}\nshadow main{assert (== (main) 0)}\n')
            compiler = root/'bin/compiler'
            runtime='#!'+sys.executable+'\nraise SystemExit(0)\n'
            if contract:
                source.with_suffix('.cases.json').write_text(json.dumps(self.document([
                    {'name':'arity','args':[],'exit_code':2},
                    {'name':'positive','args':['accepted'],'exit_code':0},
                    {'name':'refusal','args':['rejected'],'exit_code':1,'contains':['ownership']},
                ])))
                runtime='#!'+sys.executable+'\nimport sys\nif len(sys.argv)==1:sys.exit(2)\nif sys.argv[1]=="accepted":sys.exit(0)\nprint("ownership");sys.exit(1)\n'
            compiler.write_text('#!'+sys.executable+'\nimport pathlib,sys\n'+
                f'mode={mode!r}\n'+
                'if mode=="compile_failure":print("retained compiler refusal");sys.exit(1)\n'+
                'if mode=="missing":sys.exit(0)\n'+
                'output=pathlib.Path(sys.argv[sys.argv.index("-o")+1])\n'+
                'output.write_text('+repr(runtime)+');output.chmod(0o700)\n')
            compiler.chmod(0o700)
            stale_marker=root/'stale-executed'
            old=('#!'+sys.executable+'\nfrom pathlib import Path\nPath('+repr(str(stale_marker))+').touch()\nraise SystemExit(0)\n').encode()
            if prior:
                (root/'.test_output').mkdir()
                (root/'.test_output/nl_nl_syntax_fixture.nano.out').write_bytes(old)
                (root/'.test_output/nl_nl_syntax_fixture.nano.out').chmod(0o700)
            report=root/'results.xml'
            result=subprocess.run(['bash',root/'tests/run_all_tests.sh','--lang','--format=junit',f'--output={report}'],
                cwd=root,env=dict(os.environ,NANOLANG_COMPILER=str(compiler),NANOLANG_BACKEND='c'),
                capture_output=True,text=True,timeout=15)
            xml=ET.parse(report).getroot()
            self.assertEqual(int(xml.attrib['tests']),1,result.stdout+result.stderr)
            self.assertEqual(len(xml.findall('.//testcase')),1)
            float(xml.attrib['time'])
            for node in xml.findall('.//testcase'):
                self.assertGreaterEqual(float(node.attrib['time']),0)
            self.assertFalse(stale_marker.exists())
            if contract:
                logs=list((root/'.test_output').glob('*.run.log'))
                self.assertEqual(len(logs),1)
                self.assertEqual(logs[0].read_text().count('"accepted": true'),3)
            if prior:
                files=list((root/'.test_output').glob('prior-*/*.out'))
                self.assertEqual(len(files),1)
                self.assertEqual(files[0].read_bytes(),old)
            return result,xml

    def test_first_compile_failure_has_valid_timing_and_report(self):
        result,report=self.runner('compile_failure')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(report.attrib['failures'],'1')
        self.assertNotIn('command not found',result.stderr)

    def test_success_without_publication_cannot_reuse_prior_output(self):
        result,report=self.runner('missing',prior=True)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(report.attrib['failures'],'1')

    def test_fresh_publication_preserves_prior_and_passes(self):
        result,report=self.runner('success',prior=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(report.attrib['failures'],'0')

    def test_comprehensive_report_preserves_case_and_all_cli_outcomes(self):
        result,report=self.runner('success',contract=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(report.attrib['failures'],'0')

    def test_dangling_contract_refuses_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'case.cases.json').symlink_to('missing.json')
            marker=root/'executed'
            result,log=self.execute(root,f'from pathlib import Path;Path({str(marker)!r}).touch()')
            self.assertEqual(result.returncode,1,log)
            self.assertFalse(marker.exists())
