from pathlib import Path
import os,subprocess,sys,json,time,unittest
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root);sys.path.insert(0,str(root))
out=Path('/private/tmp/nanolang-coverage-harness-probe');out.mkdir(exist_ok=True)
cc='/opt/homebrew/opt/llvm/bin/clang'
steps=[]
if not (out/'runtime.o').exists():
 objects=[]
 for name in ('dyn_array','gc','gc_struct'):
  obj=out/(name+'.o');objects.append(str(obj));args=[cc,'--coverage','-fsanitize=address,undefined','-std=c99','-g','-O0','-fPIC','-D_GNU_SOURCE','-Isrc','-c','src/runtime/'+name+'.c','-o',str(obj)]
  p=subprocess.run(args,capture_output=True,text=True);steps.append({'argv':args,'exit':p.returncode});(out/(name+'.log')).write_text(p.stdout+p.stderr)
  if p.returncode:raise SystemExit(p.returncode)
 args=[cc,'-r','-nostdlib','-o',str(out/'runtime.o'),*objects];p=subprocess.run(args,capture_output=True,text=True);steps.append({'argv':args,'exit':p.returncode});(out/'link.log').write_text(p.stdout+p.stderr)
 (out/'build.json').write_text(json.dumps(steps,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
os.environ.update(CC=cc,NANO_NATIVE_TEST_CC=cc,LDFLAGS='--coverage',NANO_BUILD_CACHE=str(root/'obj/module_cache'))
from tests import test_bootstrap_components as target
original=subprocess.run;commands=[];replacements=0
def run(args,*a,**kw):
 global replacements
 args=list(args)
 key=str(root/'bin/nano_aot_runtime.o')
 if key in args:
  args[args.index(key)]=str(out/'runtime.o');replacements+=1
 start=time.time();p=original(args,*a,**kw)
 commands.append({'argv':list(map(str,args)),'exit':p.returncode,'seconds':time.time()-start})
 return p
subprocess.run=run
mode=sys.argv[1]
with (out/(mode+'.log')).open('w') as f:
 result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([target.BootstrapComponents('test_real_emitter_driver_executes_in_vm_and_sanitized_native')]))
(out/(mode+'.json')).write_text(json.dumps({'success':result.wasSuccessful(),'runtime_object_substitutions':replacements,'commands':commands},indent=2)+'\n')
assert replacements==1
raise SystemExit(0 if result.wasSuccessful() else 1)
