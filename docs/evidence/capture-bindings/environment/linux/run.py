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
if platform.system() == 'Darwin':
    command('sdk-path', ['/usr/bin/xcrun', '--show-sdk-path'])
    command('sdk-version', ['/usr/bin/xcrun', '--show-sdk-version'])
    sdk = (REPORT / 'sdk-path.stdout').read_text().strip()
    assert Path(sdk).is_dir()
    settings = Path(sdk) / 'SDKSettings.json'
    (REPORT / 'sdk.json').write_text(json.dumps({'path': sdk, 'realpath': str(Path(sdk).resolve()), 'settings_sha256': hashlib.sha256(settings.read_bytes()).hexdigest()}, indent=2)+'\n')
    compilers = [('apple', '/usr/bin/clang', False),
                 ('brew', '/opt/homebrew/opt/llvm/bin/clang', False),
                 ('brew-sanitized', '/opt/homebrew/opt/llvm/bin/clang', True)]
else:
    compilers = [('gcc', '/usr/bin/gcc', False), ('clang', '/usr/local/bin/clang', False),
                 ('gcc-sanitized', '/usr/bin/gcc', True), ('clang-sanitized', '/usr/local/bin/clang', True)]
tools = {}
for name, compiler, sanitized in compilers:
    p = Path(compiler).resolve()
    tools[name] = {'path': compiler, 'realpath': str(p),
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    command(name + '-version', [compiler, '--version'])
(REPORT / 'tools.json').write_text(json.dumps(tools, indent=2) + '\n')
for name, compiler, sanitized in compilers:
    output = REPORT / name
    flags = ['-std=c11', '-Wall', '-Wextra', '-Werror', '-g', '-O1']
    if platform.system() != 'Darwin' and 'clang' in name:
        flags += ['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
    if platform.system() == 'Darwin':
        flags += ['-isysroot', sdk]
    if sanitized:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    command(name + '-build', [compiler, *flags, 'tests/nanovm/test_binding_closure.c',
        'src/nanovm/heap_cycles.c', 'src/nanovm/value.c', 'src/nanoisa/isa.c', '-lm', '-o', str(output)])
    command(name + '-run', [str(output)])
    command(name + '-storage-build', [compiler, *flags, 'tests/nanovm/test_binding_state.c', 'src/nanovm/heap.c', 'src/nanovm/heap_cycles.c', 'src/nanovm/value.c', 'src/nanoisa/isa.c', '-lm', '-o', str(output)+'-storage'])
    command(name + '-storage-run', [str(output)+'-storage'])
    text = (REPORT / (name + '-run.stdout')).read_text()
    assert 'atomic closure checks with complete staged-allocation recovery.' in text
after = inventory()
assert after == manifest['inputs']
(REPORT / 'inputs-after.json').write_text(json.dumps(after, indent=2) + '\n')
(REPORT / 'complete.json').write_text(json.dumps({'source': manifest['source'],
    'platform': platform.platform(), 'configurations': len(compilers), 'status': 'pass'}, indent=2) + '\n')
