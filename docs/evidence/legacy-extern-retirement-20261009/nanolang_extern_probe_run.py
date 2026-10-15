from pathlib import Path
import subprocess,os,time,json
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-extern-fixture-probe');os.chdir(root)
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
steps=[]
for tool in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 module=out/(tool+'.nvm');c=out/(tool+'.c');binary=out/(tool+'.native')
 cmds=[[str(root/'bin'/tool),str(out/'driver.nano'),'--emit-nvm','-o',str(module)],['bin/nano_vm','--verify-only',str(module)],['bin/nano_vm',str(module)],['bin/nvm2c',str(module),'-o',str(c)],['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-O0','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(binary)],[str(binary)]]
 for args in cmds:
  start=time.time();log=out/(str(len(steps))+'.log')
  with log.open('w') as f:r=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  steps.append({'argv':args,'exit':r.returncode,'seconds':time.time()-start,'log':str(log)})
  (out/'manifest.json').write_text(json.dumps(steps,indent=2)+'\n')
  print(tool,len(steps),r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
