"""I test real native gate lifecycle and constructed parser refusals, not Linux BPF execution."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from scripts import observe_cseed_kernel_exit as observer

ROOT=Path(__file__).resolve().parents[1]

class KernelExitObserver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        cls.work=Path(tempfile.mkdtemp(prefix='kernel-exit-controls-',dir=ROOT/'.test_output'))
        print('I retain native gate controls and constructed parser inputs at',cls.work,flush=True)
        cls.inputs={str(p):observer.digest(p) for p in (ROOT/'scripts/observe_cseed_kernel_exit.py',ROOT/'tests/nanoisa/test_cseed_exit_observer.c',Path(__file__).resolve(),ROOT/'docs/evidence/cseed-kernel-exit-20261109/first-linux/texts.json.gz')}
        cc=shlex.split(os.environ.get('CC','cc'))
        tool=Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
        cls.tool=(tool,observer.digest(tool))
        cls.binary=cls.work/'actual-native-control'
        command=[*cc,'-std=c99','-O2','-g','-Wall','-Wextra','-Werror','-UNDEBUG',ROOT/'tests/nanoisa/test_cseed_exit_observer.c','-o',cls.binary]
        result=subprocess.run(list(map(str,command)),cwd=ROOT,capture_output=True,timeout=30)
        (cls.work/'build.stdout').write_bytes(result.stdout);(cls.work/'build.stderr').write_bytes(result.stderr)
        assert result.returncode==0,result.stderr
        assert cls.binary.read_bytes()[:4] in (b'\xcf\xfa\xed\xfe',b'\x7fELF')
        cls.binary_before=observer.digest(cls.binary)
        result=subprocess.run([cls.binary],capture_output=True,timeout=20)
        (cls.work/'native.stdout').write_bytes(result.stdout);(cls.work/'native.stderr').write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr
        cls.record=json.loads(result.stdout)
        assert cls.record['wait_status']==7<<8 and cls.record['end_ns']>=cls.record['start_ns']
        cls.build={'argv':list(map(str,command)),'returncode':0}

    @classmethod
    def tearDownClass(cls):
        assert cls.inputs=={p:observer.digest(p) for p in cls.inputs}
        assert observer.digest(cls.tool[0])==cls.tool[1]
        assert observer.digest(cls.binary)==cls.binary_before
        observer.write(cls.work/'validation.json',{'source_inputs_before_after':cls.inputs,'compiler':{'path':str(cls.tool[0]),'sha256_before_after':cls.tool[1]},'native_control':cls.record,'native_binary_sha256_before_after':cls.binary_before,'build':cls.build,'linux_kernel_execution_qualified':False,'parser_fixture_scope':'I construct kernel records to exercise parser refusals; these are not native BPF results.'})

    def fixture(self):
        record=copy.deepcopy(self.record);root=record['pid'];child=record['child_pid'];start=record['start_ns'];end=record['end_ns']
        def event(kind,tid,stamp,value=0,pid=None):
            return {'event':'cseed-kernel-exit','kind':kind,'root_pid':root,'pid':tid if pid is None else pid,'tid':tid,'monotonic_ns':stamp,'value':value,'detail':0}
        events=[event('ready',0,start-10),event('fork',root,start+1,child),event('exit-group',child,start+2,7),event('exit-mmap-enter',child,start+3),event('exit-mmap-return',child,start+4),event('process-exit',child,start+5),event('wait-return',root,start+6,child),event('exit-group',root,end+1),event('exit-mmap-enter',root,end+2),event('exit-mmap-return',root,end+3),event('process-exit',root,end+4),event('end',0,end+5)]
        return record,events

    def parsed(self,events,stderr=''):
        return observer.parse_events('\n'.join(json.dumps(e) for e in events),self.record['pid'],stderr)

    def test_actual_native_gate_release_and_identity(self):
        with (self.work/'gated.stdout').open('wb') as out,(self.work/'gated.stderr').open('wb') as err:
            child,gate=observer.spawn_gated([self.binary],out,err,dict(os.environ))
            try:
                self.assertIsNone(child.poll())
                os.write(gate,b'1');os.close(gate);gate=None
                self.assertEqual(child.wait(timeout=20),0)
            finally:
                if gate is not None:os.close(gate)
                observer.stop_target(child)
        result=json.loads((self.work/'gated.stdout').read_text())
        self.assertEqual(result['pid'],child.pid)
        self.assertEqual(result['wait_status'],7<<8)
        self.assertEqual((self.work/'gated.stderr').read_bytes(),b'')

    def test_missing_gate_refuses_before_native_execution(self):
        with (self.work/'closed-gate.stdout').open('wb') as out,(self.work/'closed-gate.stderr').open('wb') as err:
            child,gate=observer.spawn_gated([self.binary],out,err,dict(os.environ))
            os.close(gate)
            try:self.assertNotEqual(child.wait(timeout=10),0)
            finally:observer.stop_target(child)
        self.assertEqual((self.work/'closed-gate.stdout').read_bytes(),b'')
        self.assertIn('I refuse a missing observation gate.',(self.work/'closed-gate.stderr').read_text())

    def test_interruption_reaps_a_separate_native_wrapper_session(self):
        pidfile=self.work/'separate-session.pid'
        code='from tests.native_process import run_captured; import sys; run_captured([sys.executable,"-c",'+repr('import os,time; from pathlib import Path; p=Path('+repr(str(pidfile))+'); t=p.with_suffix(".tmp"); t.write_text(str(os.getpid())); t.replace(p); time.sleep(300)')+'], timeout=300)'
        with (self.work/'interrupted.stdout').open('wb') as out,(self.work/'interrupted.stderr').open('wb') as err:
            child,gate=observer.spawn_gated([sys.executable,'-c',code],out,err,dict(os.environ))
            try:
                os.write(gate,b'1');os.close(gate);gate=None
                until=time.monotonic()+10
                while not pidfile.exists() and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(pidfile.exists())
                nested=int(pidfile.read_text())
                observer.stop_target(child)
                self.assertIsNotNone(child.poll())
                with self.assertRaises(ProcessLookupError):os.kill(nested,0)
            finally:
                if gate is not None:os.close(gate)
                observer.stop_target(child)

    def test_constructed_protocol_checks_control_clock_and_async_order(self):
        record,events=self.fixture()
        parsed=self.parsed(list(reversed(events)))
        checked=observer.check_control(parsed,record,record['pid'])
        self.assertTrue(checked['valid'])
        self.assertEqual(checked['child_exit_mmap']['wall_ns'],1)
        observer.write(self.work/'constructed-parser-fixture.json',{'scope':'I construct these events; they are not kernel execution evidence.','native_record':record,'events':events})

    def test_constructed_early_process_exit_keeps_late_teardown(self):
        record,events=self.fixture();root=record['pid'];child=record['child_pid'];start=record['start_ns']
        for event in events:
            if event['tid']==child:
                if event['kind']=='process-exit':event['monotonic_ns']=start+3
                elif event['kind']=='exit-mmap-enter':event['monotonic_ns']=start+4
                elif event['kind']=='exit-mmap-return':event['monotonic_ns']=start+5
        events.insert(-1,{'event':'cseed-kernel-exit','kind':'task-free','root_pid':root,'pid':0,'tid':child,'monotonic_ns':start+7,'value':0,'detail':root})
        self.assertTrue(observer.check_control(self.parsed(events),record,root)['valid'])
        broken=copy.deepcopy(events)
        next(e for e in broken if e['kind']=='task-free')['monotonic_ns']=start+2
        with self.assertRaises(ValueError):self.parsed(broken)

    def test_retained_actual_linux_control_still_requires_missing_teardown(self):
        with gzip.open(ROOT/'docs/evidence/cseed-kernel-exit-20261109/first-linux/texts.json.gz','rt') as stream:texts=json.load(stream)
        prefix='.test_output/cseed-kernel-exit/control/'
        record=json.loads(texts[prefix+'target.stdout'])
        events=observer.parse_events(texts[prefix+'kernel.jsonl'],record['pid'],texts[prefix+'observer.stderr'])
        with self.assertRaisesRegex(ValueError,'teardown'):
            observer.check_control(events,record,record['pid'])

    def test_constructed_protocol_refuses_tampered_or_incomplete_records(self):
        record,events=self.fixture()
        cases=[]
        d=copy.deepcopy(events);d[3]['root_pid']+=1;cases.append(d)
        d=copy.deepcopy(events);d[3]['monotonic_ns']=True;cases.append(d)
        d=copy.deepcopy(events);d[3]['pid']=0;cases.append(d)
        d=copy.deepcopy(events);del d[1];cases.append(d)
        d=copy.deepcopy(events);del d[-2];cases.append(d)
        d=copy.deepcopy(events);d.insert(4,copy.deepcopy(d[3]));cases.append(d)
        d=copy.deepcopy(events);del d[-1];cases.append(d)
        for changed in cases:
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.parsed(changed)
        with self.assertRaises(ValueError):observer.parse_events('{"event":',record['pid'])
        with self.assertRaises(ValueError):self.parsed(events,'Lost 1 events')
        bad=copy.deepcopy(events);del bad[4]
        with self.assertRaises(ValueError):observer.check_control(self.parsed(bad),record,record['pid'])
        bad=copy.deepcopy(record);bad['start_ns']=False
        with self.assertRaises(ValueError):observer.check_control(self.parsed(events),bad,record['pid'])
        bad=copy.deepcopy(record);bad['end_ns']=bad['start_ns']
        with self.assertRaises(ValueError):observer.check_control(self.parsed(events),bad,record['pid'])

    def test_full_retained_factory_capture_replays_with_bounded_parser(self):
        capture=ROOT/'docs/evidence/cseed-kernel-exit-20261109/corrected-linux/kernel.jsonl.gz'
        lifecycle=ROOT/'docs/evidence/cseed-kernel-exit-20261109/corrected-linux/factory-lifecycle.json'
        code='import gzip,json; from scripts.observe_cseed_kernel_exit import parse_events; '+\
             'record=json.load(open('+repr(str(lifecycle))+')); '+\
             'events=parse_events(gzip.open('+repr(str(capture))+',"rt").read(),record["target_pid"]); '+\
             'assert len(events)==284752; print(len(events))'
        before={str(p):observer.digest(p) for p in (capture,lifecycle)}
        started=time.monotonic()
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,timeout=30)
        elapsed=time.monotonic()-started
        self.assertEqual(before,{p:observer.digest(p) for p in before})
        observer.write(self.work/'actual-capture-replay.json',{'inputs_sha256_before_after':before,'records':284752,'wall_seconds':elapsed,'parser_timeout_seconds':30,'new_kernel_capture_performed':False})
        (self.work/'actual-capture-replay.stdout').write_bytes(result.stdout)
        (self.work/'actual-capture-replay.stderr').write_bytes(result.stderr)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        self.assertEqual(result.stdout,b'284752\n')
        self.assertEqual(result.stderr,b'')

    def test_render_refuses_non_pid_inputs_and_uses_explicit_clock(self):
        for value in (True,0,-1,'1; exit();'):
            with self.assertRaises(ValueError):observer.render_program(value)
        program=observer.render_program(self.record['pid'])
        self.assertIn('nsecs(monotonic)',program)
        self.assertNotIn('override(',program)
        self.assertNotIn('signal(',program)

if __name__=='__main__':unittest.main()
