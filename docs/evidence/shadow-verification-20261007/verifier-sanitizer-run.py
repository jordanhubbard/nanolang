from pathlib import Path
import os, shlex, subprocess
out = Path('/tmp/nanolang-5.1-20261007')
clang = '/opt/homebrew/opt/llvm/bin/clang'
flags = ['-O1', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
def recipe(target, source=None):
    args = ['make', '-n'] + (['-W', source] if source else []) + [target]
    return [shlex.split(line) for line in subprocess.check_output(args, text=True).replace('\\\n', ' ').splitlines()]
replacements = {}
for source, obj, name in [('src/nanoisa/verifier.c', 'obj/nanoisa/verifier.o', 'verifier'), ('modules/nanoisa/nanoisa.c', 'obj/nanoisa/nanoisa_facade.o', 'facade')]:
    cmd = next(c for c in recipe(obj, source) if '-c' in c and source in c)
    dest = str(out / ('verifier-san-' + name + '.o'))
    replacements[obj] = dest
    cmd = [dest if x == obj else x for x in cmd]
    cmd[0] = clang
    subprocess.run(cmd + flags, check=True)
with (out / 'shadow-verifier-sanitized-final.log').open('w') as log:
    for target, binary in [('test-verifier', 'tests/nanoisa/test_verifier'), ('test-nvm-v2-endtoend', 'tests/nanoisa/test_nvm_v2_endtoend')]:
        cmd = next(c for c in recipe(target) if '-o' in c and binary in c)
        dest = str(out / ('san-' + Path(binary).name))
        cmd = [dest if x == binary else replacements.get(x, x) for x in cmd]
        cmd[0] = clang
        subprocess.run(cmd + flags, check=True)
        r = subprocess.run([dest], capture_output=True, text=True, env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'})
        log.write(target + ' exit=' + str(r.returncode) + '\n' + r.stdout + r.stderr)
        log.flush()
        print(target, r.returncode, flush=True)
        if r.returncode: raise SystemExit(r.returncode)
