#!/usr/bin/env python3
"""I observe one original source factory's Linux exit boundaries without changing supervision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE_EQUIVALENT='35a32bf746e9c7741158bb8e62563d92bde614f8'
KINDS={'ready','end','fork','exit-group','exit-thread','exit-mmap-enter','exit-mmap-return','process-exit','kill','wait-return'}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_program(root_pid):
    if type(root_pid) is not int or root_pid<=0:
        raise ValueError('I require a positive actual target PID.')
    def emit(kind,value='0',detail='0',pid='pid',tid='tid'):
        fmt='{"event":"cseed-kernel-exit","kind":"'+kind+'","root_pid":'+str(root_pid)+',"pid":%llu,"tid":%llu,"monotonic_ns":%llu,"value":%lld,"detail":%lld}\n'
        return 'printf('+json.dumps(fmt)+', (uint64)'+pid+', (uint64)'+tid+', nsecs(monotonic), (int64)('+value+'), (int64)('+detail+'));'
    return '\n'.join([
        'BEGIN { @tracked['+str(root_pid)+'] = 1; '+emit('ready',pid='0',tid='0')+' }',
        'tracepoint:sched:sched_process_fork /@tracked[args->parent_pid]/ { @tracked[args->child_pid] = 1; '+emit('fork','args->child_pid')+' }',
        'tracepoint:syscalls:sys_enter_exit_group /@tracked[tid]/ { '+emit('exit-group','args->error_code')+' }',
        'tracepoint:syscalls:sys_enter_exit /@tracked[tid]/ { '+emit('exit-thread','args->error_code')+' }',
        'kprobe:exit_mmap /@tracked[tid]/ { '+emit('exit-mmap-enter')+' }',
        'kretprobe:exit_mmap /@tracked[tid]/ { '+emit('exit-mmap-return')+' }',
        'tracepoint:sched:sched_process_exit /@tracked[tid]/ { '+emit('process-exit')+' delete(@tracked[tid]); }',
        'tracepoint:syscalls:sys_enter_kill /@tracked[tid]/ { '+emit('kill','args->pid','args->sig')+' }',
        'tracepoint:syscalls:sys_exit_wait4 /@tracked[tid] && args->ret != 0/ { '+emit('wait-return','args->ret')+' }',
        'END { '+emit('end',pid='0',tid='0')+' clear(@tracked); }',''])

def parse_events(text,root_pid,stderr=''):
    if re.search(r'\b(lost|dropped|error|failed|unable)\b|\bcould not\b',stderr,re.I):
        raise ValueError('I refuse observer errors or lost events.')
    events=[]
    for line in text.splitlines():
        if not line.strip(): continue
        event=json.loads(line)
        if set(event)!={'event','kind','root_pid','pid','tid','monotonic_ns','value','detail'}:
            raise ValueError('I refuse an incomplete kernel record.')
        if event['event']!='cseed-kernel-exit' or event['kind'] not in KINDS:
            raise ValueError('I refuse an unknown kernel event.')
        for name in ('root_pid','pid','tid','monotonic_ns','value','detail'):
            if type(event[name]) is not int: raise ValueError('I require integer kernel event fields.')
        if event['root_pid']!=root_pid or event['monotonic_ns']<0:
            raise ValueError('I refuse a changed observer root or clock.')
        if event in events: raise ValueError('I refuse duplicate kernel records.')
        events.append(event)
    events.sort(key=lambda e:e['monotonic_ns'])
    if not events or events[0]['kind']!='ready' or events[-1]['kind']!='end':
        raise ValueError('I require complete observer start and stop boundaries.')
    if sum(e['kind']=='ready' for e in events)!=1 or sum(e['kind']=='end' for e in events)!=1:
        raise ValueError('I require one observer start and stop.')
    active={root_pid}
    for event in events:
        if event['kind'] in ('ready','end'):
            if event['pid'] or event['tid']: raise ValueError('I refuse a changed observer boundary identity.')
            continue
        if event['pid']<=0 or event['tid'] not in active:
            raise ValueError('I refuse an event outside the tracked target family.')
        if event['kind']=='fork':
            if event['value']<=0 or event['value'] in active:
                raise ValueError('I refuse an inconsistent child identity.')
            active.add(event['value'])
        elif event['kind']=='process-exit': active.remove(event['tid'])
    if active: raise ValueError('I require terminal events for the complete observed family.')
    return events

def exit_intervals(events,tid):
    entered=None;intervals=[]
    for event in events:
        if event['tid']!=tid: continue
        if event['kind']=='exit-mmap-enter':
            if entered is not None: raise ValueError('I refuse nested address-space teardown observations.')
            entered=event['monotonic_ns']
        elif event['kind']=='exit-mmap-return':
            if entered is None: raise ValueError('I refuse an unmatched teardown return.')
            intervals.append({'start_ns':entered,'end_ns':event['monotonic_ns'],'wall_ns':event['monotonic_ns']-entered})
            entered=None
    if entered is not None: raise ValueError('I refuse an incomplete teardown interval.')
    return intervals

def check_control(events,record,root_pid):
    if set(record)!={'pid','child_pid','wait_status','start_ns','end_ns'} or any(type(v) is not int for v in record.values()):
        raise ValueError('I require complete integer native control fields.')
    if record['child_pid']<=0 or record['start_ns']<0:
        raise ValueError('I refuse an invalid native control identity or clock.')
    if record['pid']!=root_pid or record['wait_status']!=7<<8:
        raise ValueError('I refuse the wrong actual native control result.')
    child=record['child_pid']
    entries=[e for e in events if e['tid']==child and e['pid']==child and e['kind']=='exit-group']
    exits=[e for e in events if e['tid']==child and e['kind']=='process-exit']
    waits=[e for e in events if e['tid']==root_pid and e['pid']==root_pid and e['kind']=='wait-return' and e['value']==child]
    intervals=exit_intervals(events,child)
    if len(entries)!=1 or entries[0]['value']!=7 or len(exits)!=1 or len(waits)!=1 or len(intervals)!=1:
        raise ValueError('I require the actual child exit, teardown and parent reap witnesses.')
    points=[record['start_ns'],entries[0]['monotonic_ns'],intervals[0]['start_ns'],intervals[0]['end_ns'],exits[0]['monotonic_ns'],waits[0]['monotonic_ns'],record['end_ns']]
    if points!=sorted(points): raise ValueError('I refuse inconsistent native and kernel monotonic clocks.')
    return {'valid':True,'native':record,'child_exit_mmap':intervals[0],'clock':'CLOCK_MONOTONIC','kernel_execution_observed':True}

def gated(fd,command):
    try:
        if os.read(fd,1)!=b'1': raise RuntimeError('I refuse a missing observation gate.')
    finally: os.close(fd)
    os.execvpe(command[0],command,os.environ)

def spawn_gated(command,stdout,stderr,env):
    reader,writer=os.pipe()
    try:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'gated','--fd',str(reader),'--',*map(str,command)],cwd=ROOT,env=env,pass_fds=(reader,),start_new_session=True,stdout=stdout,stderr=stderr)
    except BaseException:
        os.close(writer);raise
    finally:os.close(reader)
    return child,writer

def stop_target(child):
    if child.poll() is not None: return
    try:os.killpg(child.pid,signal.SIGINT)
    except ProcessLookupError:return
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)

def stop_observer(observer):
    for sig in ('INT','TERM','KILL'):
        if observer.poll() is not None:return
        subprocess.run(['sudo','-n','/bin/kill','-'+sig,'--','-'+str(observer.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=10,check=True)
        try:observer.wait(timeout=10 if sig=='INT' else 5)
        except subprocess.TimeoutExpired:continue
    if observer.poll() is None:raise RuntimeError('I cannot reap my owned kernel observer.')

def observe(command,output,timeout,env):
    output.mkdir(parents=True,exist_ok=False)
    observer=None;child=None;gate=None;record={}
    try:
        with (output/'target.stdout').open('wb') as stdout,(output/'target.stderr').open('wb') as stderr,(output/'kernel.jsonl').open('wb') as kernel,(output/'observer.stderr').open('wb') as observer_error:
            child,gate=spawn_gated(command,stdout,stderr,env)
            record={'target_pid':child.pid,'target_argv':list(map(str,command)),'target_uid':os.getuid(),'target_started':False,'timeout_seconds':timeout}
            program=output/'observer.bt';program.write_text(render_program(child.pid))
            record['probe_program_sha256']=digest(program)
            observer=subprocess.Popen(['sudo','-n',str(Path(shutil.which('bpftrace')).resolve(strict=True)),'-q','-B','line',str(program)],stdout=kernel,stderr=observer_error,start_new_session=True)
            record['observer_pid']=observer.pid
            until=time.monotonic()+20
            ready=False
            while time.monotonic()<until:
                if observer.poll() is not None:break
                for line in (output/'kernel.jsonl').read_text().splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    if event.get('kind')=='ready' and event.get('root_pid')==child.pid:ready=True
                if ready:break
                time.sleep(.05)
            if not ready:raise RuntimeError('I require attached kernel probes before releasing the original target.')
            os.write(gate,b'1');os.close(gate);gate=None
            record['target_started']=True
            record['target_returncode']=child.wait(timeout=timeout)
    finally:
        if gate is not None:os.close(gate)
        if child is not None:stop_target(child);record['target_final_returncode']=child.returncode
        if observer is not None:stop_observer(observer);record['observer_returncode']=observer.returncode
        write(output/'lifecycle.json',record)
    if digest(output/'observer.bt')!=record['probe_program_sha256']:raise RuntimeError('I refuse changed probe-program bytes.')
    if record['observer_returncode']!=0:raise RuntimeError('I require a successful observer shutdown.')
    events=parse_events((output/'kernel.jsonl').read_text(),child.pid,(output/'observer.stderr').read_text())
    write(output/'events.json',events)
    return record,events

def preflight(output):
    if sys.platform!='linux':raise RuntimeError('I require Linux for actual kernel observation.')
    operations=[]
    bpf=str(Path(shutil.which('bpftrace')).resolve(strict=True))
    write(output/'observer-tool.json',{'path':bpf,'sha256_before':digest(bpf)})
    commands=[['uname','-srvmo'],[bpf,'--version'],['sudo','-n','/usr/bin/timeout','--kill-after=2s','15s',bpf,'--info']]
    commands += [['sudo','-n','/usr/bin/timeout','--kill-after=2s','15s',bpf,'-l',probe] for probe in ('kprobe:exit_mmap','kretprobe:exit_mmap','tracepoint:sched:sched_process_fork','tracepoint:sched:sched_process_exit','tracepoint:syscalls:sys_enter_exit_group','tracepoint:syscalls:sys_enter_exit','tracepoint:syscalls:sys_enter_kill','tracepoint:syscalls:sys_exit_wait4')]
    for i,command in enumerate(commands):
        result=subprocess.run(command,capture_output=True,text=True,timeout=20)
        operations.append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        write(output/'preflight.json',operations)
        available=True
        if i>=3:
            requested=command[-1]
            names={line.strip().replace(':vmlinux:',':') for line in result.stdout.splitlines()}
            available=requested in names
        if result.returncode or not available:raise RuntimeError('I require each declared kernel probe to be available.')
    return operations

def factory_env():
    if os.environ.get('NANO_SHADOW_TIMEOUT_SECONDS') not in (None,'10'):
        raise ValueError('I preserve the original ten-second shadow deadline.')
    return dict(os.environ,NANO_SHADOW_TRACE='1',NANO_SHADOW_PHASE_TRACE='1',NANO_NVM2C_SHAPE_TRACE='1',
        NANO_STRING_OPERAND_COMPILERS='nanoc_c,nano_virt',NANO_PRINT_STATEMENT_COMPILERS='nanoc_c,nano_virt',
        NANO_EMPTY_RECORD_PROVIDERS='nanoc_c,nano_virt',NANO_NUMERIC_PROVIDERS='nano_virt',NANO_MAP_CONTEXT_PROVIDERS='nanoc_c,nano_virt',
        NANOLANG_SELFHOST_COMPILER=str(ROOT/'bin/nano_virt'),NANO_NATIVE_TEST_CC=os.environ.get('CC','gcc'))

def run(output):
    from scripts.compare_cseed_source_creation import FACTORY_FILES,PROVIDERS,health,shared_inputs,shadow_phases,shadow_trace
    from tests.native_process import native_compile_timeout
    output.mkdir(parents=True,exist_ok=False)
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(output/'plan.json',{'source_pin':pin,'product_source_equivalent_to':SOURCE_EQUIVALENT,'actual_factory_attempts':1,'shadow_deadline_seconds':10,'factory_native_budget_seconds':native_compile_timeout(),'clock':'CLOCK_MONOTONIC','target_uid':os.getuid(),'requires_real_native_observer_control':True,'production_changed':False,'release_qualified':False,'scope':'I observe exit entry, address-space teardown, exit and reap boundaries. Instrumentation perturbs timing and cannot retrospectively assign a previous failure cause or establish a whole-factory gain.'})
    health(output,'before-observer')
    subprocess.run(['git','diff','--exit-code',SOURCE_EQUIVALENT,'HEAD','--','src','src_nano','modules','Makefile.gnu','GNUmakefile','Makefile',*FACTORY_FILES],cwd=ROOT,check=True,timeout=30)
    preflight(output)
    observer_tool=json.loads((output/'observer-tool.json').read_text())
    observer_sources={path:digest(ROOT/path) for path in ('scripts/observe_cseed_kernel_exit.py','tests/nanoisa/test_cseed_exit_observer.c','tests/test_cseed_kernel_exit.py','.github/workflows/cseed-kernel-exit.yml')}
    for path,value in observer_sources.items():
        if hashlib.sha256(subprocess.check_output(['git','show',pin+':'+path],cwd=ROOT)).hexdigest()!=value:
            raise RuntimeError('I refuse observer inputs different from the exact published source.')
    write(output/'observer-sources-before.json',observer_sources)
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','src_nano','modules','Makefile.gnu','GNUmakefile','Makefile',*FACTORY_FILES],cwd=ROOT,check=True,timeout=30)
    before=shared_inputs(ROOT);providers={p:digest(ROOT/p) for p in PROVIDERS}
    for path,value in before.items():
        for revision in (pin,SOURCE_EQUIVALENT):
            if hashlib.sha256(subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT)).hexdigest()!=value:
                raise RuntimeError('I require every factory source input to match both immutable Git pins.')
    write(output/'sources-before.json',before);write(output/'providers-before.json',providers)
    cc=shlex.split(os.environ.get('CC','gcc'));tool=Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
    tool_before=digest(tool)
    write(output/'compiler.json',{'argv':cc,'path':str(tool),'sha256':tool_before,'version':subprocess.check_output([*cc,'--version'],text=True,timeout=10)})
    control=output/'native-control'
    argv=[*cc,'-std=c99','-O2','-g','-Wall','-Wextra','-Werror','-UNDEBUG',str(ROOT/'tests/nanoisa/test_cseed_exit_observer.c'),'-o',str(control)]
    compiled=subprocess.run(argv,capture_output=True,timeout=30)
    (output/'control-build.stdout').write_bytes(compiled.stdout);(output/'control-build.stderr').write_bytes(compiled.stderr)
    write(output/'control-build.json',{'argv':argv,'returncode':compiled.returncode})
    if compiled.returncode:raise RuntimeError('I require my actual native observer control to build.')
    data=control.read_bytes()
    if data[:4]!=b'\x7fELF' or struct.unpack_from('<H',data,16)[0] not in (2,3):raise RuntimeError('I require an actual Linux native control image.')
    control_before=digest(control)
    lifecycle,events=observe([control],output/'control',30,dict(os.environ))
    if lifecycle['target_returncode']!=0:raise RuntimeError('I require the actual native control to execute successfully.')
    validation=check_control(events,json.loads((output/'control/target.stdout').read_text()),lifecycle['target_pid'])
    if digest(control)!=control_before:raise RuntimeError('I refuse a changed native control image.')
    write(output/'native-control.json',{'sha256_before_after':control_before,**validation})
    if digest(observer_tool['path'])!=observer_tool['sha256_before']:raise RuntimeError('I refuse a changed observer tool.')
    health(output,'before-one-factory')
    factory=ROOT/'.test_output/bstring-source-compiler';prior=set(factory.iterdir()) if factory.exists() else set()
    lifecycle,events=observe([sys.executable,ROOT/'scripts/compare_cseed_source_creation.py','sample',ROOT],output/'factory',native_compile_timeout()+60,factory_env())
    created=set(factory.iterdir())-prior
    if len(created)!=1:raise RuntimeError('I require exactly one original factory attempt.')
    actual=created.pop();copied=output/'actual-factory';shutil.copytree(actual,copied)
    identity=json.loads((copied/'inputs.json').read_text())
    if identity['before']!=identity['after'] or before!=shared_inputs(ROOT):raise RuntimeError('I refuse changed factory source inputs.')
    after={p:digest(ROOT/p) for p in PROVIDERS}
    write(output/'providers-after.json',after);write(output/'sources-after.json',shared_inputs(ROOT))
    observer_tool['sha256_after']=digest(observer_tool['path']);write(output/'observer-tool.json',observer_tool)
    observer_after={path:digest(ROOT/path) for path in observer_sources};write(output/'observer-sources-after.json',observer_after)
    if observer_after!=observer_sources or observer_tool['sha256_before']!=observer_tool['sha256_after']:raise RuntimeError('I refuse changed observer inputs or executable bytes.')
    if providers!=after or digest(tool)!=tool_before:raise RuntimeError('I refuse changed providers or host compiler.')
    for path in actual.rglob('*'):
        if path.is_file() and digest(path)!=digest(copied/path.relative_to(actual)):raise RuntimeError('I refuse changed copied factory evidence.')
    succeeded=lifecycle['target_returncode']==0
    raw=(copied/'source-build.stderr').read_text();phases=shadow_phases(raw,succeeded,10)
    if not phases['valid']:raise RuntimeError('I require valid original user-space phase observations.')
    status=phases['events'][-1];parent=status['pid'];child=status['child_pid']
    if not any(e['kind']=='fork' and e['pid']==parent and e['value']==child for e in events):raise RuntimeError('I require the actual source supervisor/child kernel identity.')
    intervals=exit_intervals(events,child)
    exits=[e for e in events if e['tid']==child and e['kind']=='process-exit']
    waits=[e for e in events if e['tid']==parent and e['kind']=='wait-return' and e['value']==child]
    if len(intervals)!=1 or len(exits)!=1 or len(waits)!=1:raise RuntimeError('I require complete actual child teardown, exit and reap observations.')
    origin=status['parent_start_seconds']*1000000000+status['parent_start_nanoseconds']
    relevant=[{**e,'seconds_from_original_parent_start':(e['monotonic_ns']-origin)/1e9} for e in events if e['tid']==child or e['tid']==parent and (e['kind'] in ('wait-return','kill') or e['kind']=='fork' and e['value']==child)]
    report={'source_pin':pin,'creation_succeeded':succeeded,'actual_factory_attempts':1,'phase_observation':phases,'trace':shadow_trace(raw,ROOT),'native_control':validation,'child_exit_mmap':intervals[0],'kernel_boundaries':relevant,'historical_cause_established':False,'whole_factory_gain_established':False,'native_sdk_closure_qualified':False,'release_qualified':False}
    write(output/'report.json',report)
    print(json.dumps({'creation_succeeded':succeeded,'child_exit_mmap_wall_ns':intervals[0]['wall_ns'],'observed_child_pid':child}),flush=True)
    return int(not succeeded)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    child=sub.add_parser('gated');child.add_argument('--fd',type=int,required=True);child.add_argument('command',nargs=argparse.REMAINDER)
    parent=sub.add_parser('run');parent.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='gated':
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        if not command:raise ValueError('I require an actual target command.')
        gated(args.fd,command);return 0
    def interrupted(_signal,_frame):
        raise KeyboardInterrupt('I stop my owned observer and target after interruption.')
    signal.signal(signal.SIGTERM,interrupted)
    try:return run(args.output.resolve())
    except BaseException as error:
        write(args.output.resolve()/'observation-refusal.json',{'exception':type(error).__name__,'message':str(error),'observation_qualified':False,'release_qualified':False})
        raise

if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    raise SystemExit(main())
