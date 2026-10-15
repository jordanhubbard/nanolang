from pathlib import Path
import os,shlex,subprocess
root=Path('/Users/jordanh/Src/nanolang');os.chdir(root)
lines=Path('/private/tmp/nl51-websocket-runtime-clang.log').read_text().splitlines()
commands=[shlex.split(l) for l in lines if l.startswith('/opt/homebrew/opt/llvm/bin/clang ') and 'test_websocket_runtime.c' in l]
assert len(commands)==2
for name,cc,extra in [('gcc','/opt/homebrew/bin/gcc-16',[]),('sanitized','/opt/homebrew/opt/llvm/bin/clang',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
    dest=Path('/private/tmp/nl51-websocket-runtime-'+name);dest.mkdir(exist_ok=True)
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    with (dest/'result.log').open('w') as log:
        def run(cmd,env=env):
            log.write(shlex.join(cmd)+'\n');log.flush()
            p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            log.write('exit='+str(p.returncode)+'\n');log.flush();assert p.returncode==0
        for original in commands:
            cmd=[cc,*[x for x in original[1:] if x not in ['-O3','-std=c99']],'-O1',*extra]
            i=cmd.index('-o')+1;cmd[i]=str(dest/Path(cmd[i]).name);run(cmd)
            run(['python3','-m','unittest','-v','tests.test_websocket_runtime'],dict(env,NANO_WEBSOCKET_RUNTIME=cmd[i]))
        cmd=[cc,'-std=c11','-D_GNU_SOURCE','-Wall','-Wextra','-Werror','-g','-O1',*extra,'tests/test_nsi_websocket_values.c','-o',str(dest/'value-controls')]
        run(cmd);run([str(dest/'value-controls')])
    print(name+' passed',flush=True)
