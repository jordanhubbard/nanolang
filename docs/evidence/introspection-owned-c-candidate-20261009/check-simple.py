from pathlib import Path
import subprocess,os
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-introspection-owned-c-20261009')
source=Path('/private/tmp/nanolang-introspection-routes-20261009/owned_direct.nano')
commands=[[out/'nano_virt',source,'--emit-nvm','-o',out/'program.nvm'],[root/'bin/nano_vm','--verify-only',out/'program.nvm'],[root/'bin/nano_vm',out/'program.nvm'],[root/'bin/nvm2c',out/'program.nvm','-o',out/'program.c'],['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',out/'program.c',root/'bin/nano_aot_runtime.o','-lm','-o',out/'program'],[out/'program']]
for cmd in commands:
 r=subprocess.run(list(map(str,cmd)),cwd=root,capture_output=True,text=True,timeout=120,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1'))
 print('COMMAND',list(map(str,cmd)),'EXIT',r.returncode,r.stdout,r.stderr,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
