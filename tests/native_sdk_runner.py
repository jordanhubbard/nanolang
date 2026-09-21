"""I retain bounded SDK commands outside a checkout without buffered output loss."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def process_rows():
    rows={}
    output=subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,lstart=,args='],text=True,timeout=5)
    for line in output.splitlines():
        fields=line.strip().split(None,8)
        if len(fields)==9:
            rows[int(fields[0])]=dict(parent=int(fields[1]),group=int(fields[2]),
                                     started=' '.join(fields[3:8]),command=fields[8])
    return rows


def run(directory, name, argv, cwd, extra=None, expected=(0,), timeout=180, track_descendants=False):
    directory = Path(directory)
    env = dict(os.environ, LSAN_OPTIONS='', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    argv = list(map(str, argv))
    record = {'argv': argv, 'cwd': str(cwd), 'expected': list(expected),
              'environment': {k: env.get(k) for k in ('PATH','CC','NANO_CC','TMPDIR','NANOLANG_SDK_ROOT',
                  'NANO_BUILD_CACHE','NANO_MODULE_PATH','NANO_SHADOW_TRACE','NANOVMD_SOCKET','SDK_CC_LOG','SDK_REAL_CC','SDK_REFUSED_SOURCE','SDK_REFUSAL_HIT','ASAN_OPTIONS','LSAN_OPTIONS','UBSAN_OPTIONS')}}
    (directory / (name + '-command.json')).write_text(json.dumps(record, indent=2) + '\n')
    minimum=int(env.get('NANO_SDK_MIN_FREE_BYTES','0'))
    free=shutil.disk_usage(directory).free
    (directory/(name+'-capacity.json')).write_text(json.dumps(dict(free_bytes=free,minimum_bytes=minimum))+'\n')
    if free<minimum:
        (directory/(name+'-status.json')).write_text(json.dumps(dict(first_terminal='capacity_guard',returncode=None,free_bytes=free,minimum_bytes=minimum))+'\n')
        raise AssertionError('I retain evidence and stop below the SDK command capacity guard')
    state = {'returncode': None, 'timeout': False, 'cleanup_errors': [], 'group_absent': False}
    status_file = directory / (name + '-status.json')
    def retain():
        status_file.write_text(json.dumps(state, indent=2) + '\n')
    process = None
    owned = {}
    def observe():
        rows=process_rows();descendants=set()
        if process.poll() is None:descendants.add(process.pid)
        descendants.update(pid for pid,row in rows.items() if pid in owned and row['started']==owned[pid]['started'])
        for _ in range(len(rows)+1):
            more={pid for pid,row in rows.items() if row['parent'] in descendants}
            if more<=descendants:break
            descendants.update(more)
        for pid in descendants:
            if pid in rows:owned[pid]=rows[pid]
        return rows
    start = time.monotonic()
    retain()
    stdout_path = directory / (name + '.stdout'); stderr_path = directory / (name + '.stderr')
    with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                if track_descendants:
                    deadline=time.monotonic()+timeout
                    while process.poll() is None:
                        observe()
                        remaining=deadline-time.monotonic()
                        if remaining<=0:raise subprocess.TimeoutExpired(argv,timeout)
                        try:process.wait(timeout=min(.2,remaining))
                        except subprocess.TimeoutExpired:pass
                    state['returncode']=process.returncode
                else:
                    state['returncode'] = process.wait(timeout=timeout)
                state['first_terminal'] = 'exit'
            except subprocess.TimeoutExpired:
                state.update(returncode=124, timeout=True, first_terminal='timeout')
            retain()
        except (OSError, subprocess.SubprocessError) as failure:
            state.update(first_terminal='os_error', error=repr(failure))
            retain()
        finally:
            if process is not None:
                if track_descendants:
                    try:observe()
                    except (OSError,subprocess.SubprocessError) as failure:
                        state['cleanup_errors'].append('descendant inventory: '+repr(failure))
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    state['remaining_group_killed'] = True
                except ProcessLookupError:
                    state['remaining_group_killed'] = False
                except OSError as failure:
                    state['cleanup_errors'].append(repr(failure))
                try:
                    state['child_returncode'] = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    state['cleanup_errors'].append('bounded child cleanup expired')
                deadline = time.monotonic() + 5
                while True:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        state['group_absent'] = True
                        break
                    except OSError as failure:
                        state['cleanup_errors'].append(repr(failure)); break
                    if time.monotonic() >= deadline:
                        state['cleanup_errors'].append('remaining process group'); break
                    time.sleep(.02)
            if track_descendants and process is not None:
                cleanup=[];remaining=[]
                try:
                    for sig in (signal.SIGTERM,signal.SIGKILL):
                        rows=observe()
                        groups={row['group'] for pid,row in rows.items()
                                if pid in owned and row['started']==owned[pid]['started'] and row['group']!=process.pid}
                        for group in groups:
                            try:os.killpg(group,sig);cleanup.append(dict(group=group,signal=sig.name))
                            except ProcessLookupError:pass
                        if groups:time.sleep(.2)
                    deadline=time.monotonic()+5
                    while True:
                        rows=observe()
                        remaining=[dict(pid=pid,**row) for pid,row in rows.items()
                                   if pid in owned and row['started']==owned[pid]['started'] and pid!=process.pid]
                        if not remaining or time.monotonic()>=deadline:break
                        time.sleep(.05)
                    state['descendants_absent']=not remaining
                    if remaining:state['cleanup_errors'].append('nested processes remain after bounded cleanup')
                except (OSError,subprocess.SubprocessError) as failure:
                    state['descendants_absent']=False
                    state['cleanup_errors'].append('nested cleanup: '+repr(failure))
                (directory/(name+'-descendants.json')).write_text(json.dumps(dict(
                    owned=owned,cleanup=cleanup,remaining=remaining),indent=2)+'\n')
            out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
            state['seconds'] = time.monotonic() - start
            retain()
    out = stdout_path.read_bytes(); err = stderr_path.read_bytes()
    if (state['returncode'] not in expected or state['timeout'] or
            state['cleanup_errors'] or not state['group_absent']):
        raise AssertionError((name, state, out[-8000:], err[-8000:]))
    return out, err, state
