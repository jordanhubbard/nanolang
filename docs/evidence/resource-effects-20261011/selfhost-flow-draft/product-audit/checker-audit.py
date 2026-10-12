from pathlib import Path
import sys,subprocess,json,time
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import HOST_RUNTIME,OneIrCompiler
root=Path.cwd();work=Path('/private/tmp/nl51-selfhost-resource-effects/product-audit')
steps=[]
def run(args,env=None):
 import os
 start=time.monotonic();index=len(steps)
 with (work/f'checker-{index:02d}.log').open('w') as log:
  result=subprocess.run(list(map(str,args)),cwd=root,env={**os.environ,**(env or {})},stdout=log,stderr=subprocess.STDOUT,timeout=900)
 steps.append({'argv':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-start})
 (work/'checker-manifest.json').write_text(json.dumps(steps,indent=2)+'\n')
 print(index,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run([work/'seed',root/'tests/test_resource_effect_checker.nano','--emit-nvm','-o',work/'checker.nvm'])
run([root/'bin/nano_vm','--verify-only',work/'checker.nvm'])
run([root/'bin/nvm2c',work/'checker.nvm','-o',work/'checker.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'checker.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'checker.c','-o',work/'checker',*HOST_RUNTIME])
run([sys.executable,'-m','unittest','tests.test_resource_effect_frontend','-v'],{'NANOLANG_RESOURCE_EFFECT_ROUTES':'selfhost-checker','NANOLANG_RESOURCE_EFFECT_NANO_CHECKER':str(work/'checker')})
