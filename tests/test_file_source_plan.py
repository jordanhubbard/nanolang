"""I retain fresh C ownership and paired C-seed/Stage1/Stage2 plan reports.

My orchestrator prepares a fresh bootstrap before this gate. I do not rebuild
or select another compiler silently. C sanitizers cover these three small
providers and C fixtures; I do not advertise Nano recoverable allocation parity.
"""
from pathlib import Path
import json
import os
import re
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
        process=None
        status={'returncode':None,'timeout':False,'cleanup_errors':[],'group_absent':False}
        status_path=cls.work/(name+'-status.json')
        stdout_path=cls.work/(name+'.stdout');stderr_path=cls.work/(name+'.stderr')
        def retain_status():
            status_path.write_text(json.dumps(status,indent=2)+'\n')
        retain_status()
        # Children write directly to retained files; cleanup never owns buffered
        # PIPE output and cannot discard it on a second timeout.
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            try:
                process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stdout,
                                         stderr=stderr,start_new_session=True)
                try:
                    status['returncode']=process.wait(timeout=timeout)
                    status['first_terminal']='exit'
                except subprocess.TimeoutExpired:
                    status['timeout']=True
                    status['returncode']=124
                    status['first_terminal']='timeout'
                retain_status()
            except OSError as failure:
                status['launch_or_wait_error']=repr(failure)
                status['first_terminal']='os_error'
                retain_status()
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid,signal.SIGKILL)
                        status['remaining_group_killed']=True
                    except ProcessLookupError:
                        status['remaining_group_killed']=False
                    except OSError as failure:
                        status['cleanup_errors'].append(repr(failure))
                    try:
                        status['child_returncode']=process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        status['cleanup_errors'].append('child wait timed out after group kill')
                        status['child_returncode']=process.poll()
                    except OSError as failure:
                        status['cleanup_errors'].append(repr(failure))
                    deadline=time.monotonic()+5
                    while True:
                        try:
                            os.killpg(process.pid,0)
                        except ProcessLookupError:
                            status['group_absent']=True
                            break
                        except OSError as failure:
                            status['cleanup_errors'].append(repr(failure))
                            break
                        if time.monotonic()>=deadline:
                            status['cleanup_errors'].append('process group remains after bounded cleanup')
                            break
                        time.sleep(0.02)
                stdout.flush();stderr.flush()
                os.fsync(stdout.fileno());os.fsync(stderr.fileno())
                retain_status()
        out=stdout_path.read_bytes();err=stderr_path.read_bytes()
        if (status.get('launch_or_wait_error') or status['timeout'] or status['returncode'] != 0
                or status['cleanup_errors'] or not status['group_absent']):
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
        expected_names={f'case_{i}' for i in range(len(cases))}
        expected_names.update({'emit','nano_extent_case','nano_budget_case',
            'fsp_string','fsp_number','file_source_catalog_view','fsp_text_ok',
            'fsp_value','fsp_equal','fsp_literal','fsp_id_ok','fsp_add',
            'fsp_failure','fsp_category','file_source_plan','text','number'})
        # I invoke actual native drivers with dependency shadows selected by default.
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            output=self.work/compiler
            args=[ROOT/'bin'/compiler,nano,'-o',output]
            shadow=self.work/(compiler+'-shadows.json')
            if compiler=='nanoc_c':args.extend(['--llm-shadow-json',shadow,'--verbose'])
            build_out,trace=self.command(compiler+'-build',args,timeout=900,
                                 extra={'NANO_SHADOW_TRACE':'1'})
            if compiler=='nanoc_c':
                report=json.loads(shadow.read_text())
                self.assertTrue(report['completed']);self.assertTrue(report['success'])
                self.assertEqual(report['test_count'],len(expected_names))
                self.assertEqual(report['failures'],[])
                raw_names=re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',
                                     build_out+b'\n'+trace,re.MULTILINE)
                selection_source='actual C-seed verbose records plus completed success JSON'
            else:
                raw_names=re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',
                                     trace,re.MULTILINE)
                selection_source='actual self-hosted NANO_SHADOW_TRACE records'
            raw_names=[name.decode('ascii') for name in raw_names]
            # I remove only the merger's documented owner prefix, never arbitrary
            # suffixes that could confuse number with fsp_number or case_1/11.
            selected=[re.sub(r'^__nano_module_+[0-9]+_', '', name) for name in raw_names]
            selection=dict(source=selection_source,raw=raw_names,normalized=selected,
                           expected=sorted(expected_names))
            (self.work/(compiler+'-selection.json')).write_text(json.dumps(selection,indent=2)+'\n')
            self.assertCountEqual(selected,expected_names)
            result,_=self.command(compiler+'-run',[output])
            self.assertEqual(result,baseline)

if __name__=='__main__':
    unittest.main()
