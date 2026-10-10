import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/Users/jordanh/Src/nanolang')
out = Path('/private/tmp/nanolang-file-parser-shadow-trace-20261008')
out.mkdir(exist_ok=False)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
user = root / 'tests/user_guide/refresh_language_pure_function.nano'
def digest():
    return hashlib.sha256(user.read_bytes()).hexdigest()
env = dict(os.environ)
env['PATH'] = '/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
env['NANO_SERVICE_PARSER_REPORT_DIR'] = str(out)
cmd = ['make', '-j2', 'test-file-service-parser', 'CC=/opt/homebrew/opt/llvm/bin/clang']
m = dict(source_commit=git('rev-parse', 'HEAD'), initial_status=git('status', '--porcelain'), user_sha256=digest(), command=cmd, PATH=env['PATH'], initial_free_bytes=shutil.disk_usage(out).free, runner_pid=os.getpid())
manifest = out / 'manifest.json'
manifest.write_text(json.dumps(m, indent=2)+'\n')
start = time.monotonic()
with (out / 'gate.log').open('w') as log:
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    m['make_pid'] = process.pid
    manifest.write_text(json.dumps(m, indent=2)+'\n')
    rc = process.wait()
m.update(exit_code=rc, elapsed_seconds=time.monotonic()-start, final_commit=git('rev-parse', 'HEAD'), final_status=git('status', '--porcelain'), final_user_sha256=digest(), final_free_bytes=shutil.disk_usage(out).free)
manifest.write_text(json.dumps(m, indent=2)+'\n')
print(json.dumps(m, indent=2), flush=True)
raise SystemExit(rc)
