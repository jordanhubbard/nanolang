from pathlib import Path
import os,subprocess,tempfile
root=Path.cwd();compilers={'cseed':[root/'bin/nano_virt'],'selfhost':[root/'bin/nano_vm',Path('/private/tmp/nanolang-mixed-integrated-compiler.nvm'),'--']}
def run(command):
 p=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=180,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
 print('COMMAND',list(map(str,command)),'EXIT',p.returncode,flush=True)
 if p.returncode:print(p.stdout,p.stderr,flush=True);raise SystemExit(p.returncode)
 return p.stdout
with tempfile.TemporaryDirectory() as directory:
 d=Path(directory)
 for producer,command in compilers.items():
  for fixture in ['source_borrow_shared.nano','source_borrow_exclusive.nano','source_borrow_order.nano','source_borrow_bool.nano']:
   source=root/'tests/nanoisa/fixtures'/fixture;module=d/'out.nvm';native=d/'out.c';binary=d/'program'
   run([*command,source,'--emit-nvm','-o',module]);run([root/'bin/nano_vm','--verify-only',module]);expected=run([root/'bin/nano_vm',module]);run([root/'bin/nvm2c',module,'-o',native]);run(['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-o',binary]);actual=run([binary]);assert expected==actual
   print('PASS',producer,fixture,flush=True)
