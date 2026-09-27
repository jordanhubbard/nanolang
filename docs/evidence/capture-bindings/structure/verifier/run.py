import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / ('reports-' + platform.system().lower())
REPORT.mkdir(exist_ok=False)
manifest = json.loads((ROOT / 'inputs.json').read_text())

def inventory():
    result = {}
    for name in manifest['inputs']:
        p = ROOT / name
        result[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'bytes': p.stat().st_size, 'mode': p.stat().st_mode & 0o777}
    return result

def command(name, argv, timeout=60):
    before = time.monotonic()
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
    timed_out = False
    with (REPORT / (name + '.stdout')).open('wb') as out, (REPORT / (name + '.stderr')).open('wb') as err:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            status = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
            status = child.returncode
        try:
            os.killpg(child.pid, 0)
            group_gone = False
        except ProcessLookupError:
            group_gone = True
    record = {'argv': argv, 'returncode': status, 'timed_out': timed_out,
              'group_gone': group_gone, 'elapsed_seconds': time.monotonic() - before}
    (REPORT / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(name, status, round(record['elapsed_seconds'], 3), flush=True)
    assert status == 0 and not timed_out and group_gone, record

assert inventory() == manifest['inputs']
assert shutil.disk_usage(ROOT).free >= 1024 ** 3
compiler = '/usr/bin/clang' if platform.system() == 'Darwin' else '/usr/bin/gcc'
for name, tool in [('compiler',compiler),('make','/usr/bin/make')]:
    p=Path(tool).resolve()
    (REPORT/(name+'-identity.json')).write_text(json.dumps({'path':tool,'realpath':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')
    command(name+'-version',[tool,'--version'])
command('ordinary-verifier',['/usr/bin/make','-j2','test-verifier','CC='+compiler],timeout=600)
after=inventory()
assert after == manifest['inputs']
(REPORT/'inputs-after.json').write_text(json.dumps(after,indent=2)+'\n')
(REPORT/'complete.json').write_text(json.dumps({'source':manifest['source'],'platform':platform.platform(),'status':'pass'},indent=2)+'\n')
