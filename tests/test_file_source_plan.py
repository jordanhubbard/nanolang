"""I retain fresh C ownership and paired C-seed/Stage1/Stage2 plan reports.

My orchestrator prepares a fresh bootstrap before this gate. I do not rebuild
or select another compiler silently. C sanitizers cover these three small
providers and C fixtures; I do not advertise Nano recoverable allocation parity.
"""
from pathlib import Path
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import unittest
from tests.file_source_plan_corpus import corpus, expected, generate_c, generate_nano

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ['src/nanoisa/file_source_plan.c', 'src/nanoisa/file_source_catalog.c',
             'src/nsi_file_plan.c']

class FileSourcePlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-file-source-plan-',
                                       dir=os.environ.get('NANO_FILE_SOURCE_REPORT_DIR')))
        print(f'I retain descriptive-plan artifacts at {cls.work}', flush=True)
        cls.cc = shlex.split(os.environ.get('NANO_FILE_SOURCE_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_FILE_SOURCE_CFLAGS', ''))
        cls.flags += ['-std=c99', '-Wall', '-Wextra', '-Werror', '-g', '-O1', '-I', str(ROOT)]
        if os.environ.get('NANO_FILE_SOURCE_SANITIZERS', '0') == '1':
            cls.flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        cls.objects=[]
        for source in PROVIDERS:
            obj=cls.work/(Path(source).stem+'.o')
            cls.command('provider-'+obj.stem,[*cls.cc,*cls.flags,'-c',source,'-o',str(obj)])
            cls.objects.append(str(obj))

    @classmethod
    def command(cls,name,args,timeout=180,extra=None):
        args=list(map(str,args))
        env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                 UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
        if extra: env.update(extra)
        (cls.work/(name+'-command.json')).write_text(json.dumps(dict(argv=args,cwd=str(ROOT),
            ASAN_OPTIONS=env['ASAN_OPTIONS'],UBSAN_OPTIONS=env['UBSAN_OPTIONS'],
            LSAN_OPTIONS=env['LSAN_OPTIONS']),indent=2)+'\n')
        process=None;out=b'';err=b'';status={'returncode':None,'timeout':False}
        try:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,start_new_session=True)
            try:
                out,err=process.communicate(timeout=timeout)
                status['returncode']=process.returncode
            except subprocess.TimeoutExpired:
                status['timeout']=True
                os.killpg(process.pid,signal.SIGKILL)
                out,err=process.communicate(timeout=10)
                status['returncode']=process.returncode
        except OSError as failure:
            status['launch_error']=str(failure)
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                    status['remaining_group_killed']=True
                except ProcessLookupError:
                    status['remaining_group_killed']=False
                deadline=time.monotonic()+5
                while True:
                    try: os.killpg(process.pid,0)
                    except ProcessLookupError:
                        status['group_absent']=True
                        break
                    if time.monotonic()>=deadline:
                        status['group_absent']=False
                        break
                    time.sleep(0.02)
            (cls.work/(name+'.stdout')).write_bytes(out)
            (cls.work/(name+'.stderr')).write_bytes(err)
            (cls.work/(name+'-status.json')).write_text(json.dumps(status,indent=2)+'\n')
        if status.get('launch_error') or status['timeout'] or status['returncode'] != 0 or not status.get('group_absent',False):
            raise AssertionError((name,status,out[-8000:],err[-8000:]))
        return out,err

    def test_c_ownership_allocation_and_exact_budget(self):
        for instrument in (False,True):
            name='ownership-instrumented' if instrument else 'ownership-linked'
            objects=self.objects[1:] if instrument else self.objects
            exe=self.work/name
            self.command(name+'-build',[*self.cc,*self.flags,
                *(['-DSOURCE_PLAN_INSTRUMENT'] if instrument else []),
                'tests/nanoisa/test_file_source_plan.c',*objects,'-o',exe])
            out,_=self.command(name+'-run',[exe])
            self.assertIn(b'PASS source C ownership ',out)

    def test_paired_full_modules_and_selected_shadows(self):
        cases=corpus()
        (self.work/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
        c=self.work/'corpus.c';nano=self.work/'corpus.nano'
        c.write_text(generate_c(cases));nano.write_text(generate_nano(cases))
        executable=self.work/'corpus-c'
        self.command('corpus-c-build',[*self.cc,*self.flags,c,*self.objects,'-o',executable])
        baseline,_=self.command('corpus-c-run',[executable])
        catalog,separator,body=baseline.partition(b'\n')
        self.assertEqual(separator,b'\n');self.assertTrue(catalog.startswith(b'CAT:file-source-catalog1;'))
        self.assertEqual(body,expected(cases))
        (self.work/'expected-rows.txt').write_bytes(expected(cases))
        # I invoke actual native drivers with dependency shadows selected by default.
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            output=self.work/compiler
            args=[ROOT/'bin'/compiler,nano,'-o',output]
            shadow=self.work/(compiler+'-shadows.json')
            if compiler=='nanoc_c':args.extend(['--llm-shadow-json',shadow])
            _,trace=self.command(compiler+'-build',args,timeout=900,
                                 extra={'NANO_SHADOW_TRACE':'1'})
            if compiler=='nanoc_c':
                report=json.loads(shadow.read_text())
                self.assertTrue(report['completed']);self.assertTrue(report['success'])
                self.assertGreaterEqual(report['test_count'],len(cases)+17)
            else:
                self.assertIn(b'fsp_text_ok',trace)
                self.assertIn(b'file_source_plan',trace)
                self.assertIn(b'case_0',trace)
                selected=[line.split()[-1] for line in trace.splitlines() if b'I am testing shadow ' in line]
                for helper in (b'text',b'number'):
                    self.assertTrue(any(name==helper or name.endswith(b'_'+helper) for name in selected), (helper,selected))
            result,_=self.command(compiler+'-run',[output])
            self.assertEqual(result,baseline)

if __name__=='__main__':
    unittest.main()
