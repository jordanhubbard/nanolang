"""I retain bounded SDK commands outside a checkout without buffered output loss."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run(directory, name, argv, cwd, extra=None, expected=(0,), timeout=180):
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
                  'NANO_BUILD_CACHE','NANO_MODULE_PATH','NANO_SHADOW_TRACE','ASAN_OPTIONS','LSAN_OPTIONS','UBSAN_OPTIONS')}}
    (directory / (name + '-command.json')).write_text(json.dumps(record, indent=2) + '\n')
    state = {'returncode': None, 'timeout': False, 'cleanup_errors': [], 'group_absent': False}
    status_file = directory / (name + '-status.json')
    def retain():
        status_file.write_text(json.dumps(state, indent=2) + '\n')
    process = None
    start = time.monotonic()
    retain()
    stdout_path = directory / (name + '.stdout'); stderr_path = directory / (name + '.stderr')
    with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                state['returncode'] = process.wait(timeout=timeout)
                state['first_terminal'] = 'exit'
            except subprocess.TimeoutExpired:
                state.update(returncode=124, timeout=True, first_terminal='timeout')
            retain()
        except OSError as failure:
            state.update(first_terminal='os_error', error=repr(failure))
            retain()
        finally:
            if process is not None:
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
            out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
            state['seconds'] = time.monotonic() - start
            retain()
    out = stdout_path.read_bytes(); err = stderr_path.read_bytes()
    if (state['returncode'] not in expected or state['timeout'] or
            state['cleanup_errors'] or not state['group_absent']):
        raise AssertionError((name, state, out[-8000:], err[-8000:]))
    return out, err, state
