from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-reduce-reconcile')
module=Path('/private/tmp/nl51-functional-product-kcfeo416/nano-selfhost-native-product-4vp861v_/compiler.nvm')
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'module':str(module),'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'steps':[]}
for name in ['src/nanoisa/nvm2c.c','Makefile.gnu']:
 manifest[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
def run(args,env=None):
 index=len(manifest['steps']);start=time.monotonic()
 with (work/f'product-{index:02d}.log').open('w') as log:
  result=subprocess.run(list(map(str,args)),cwd=root,env={**os.environ,**(env or {})},stdout=log,stderr=subprocess.STDOUT,timeout=240)
 manifest['steps'].append({'argv':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-start,'log':f'product-{index:02d}.log'})
 (work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(index,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run([root/'bin/nano_vm','--verify-only',module])
run([root/'bin/nvm2c',module,'-o',work/'compiler-final.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'compiler-final.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'compiler-final.c','-o',work/'compiler-final',*HOST_RUNTIME])
run([work/'compiler-final','--help'])
run([sys.executable,'-m','unittest','tests.test_native_nominal_order','-v'],{'NANOLANG_NOMINAL_COMPILER_ROOT':str(work),'NANOLANG_NOMINAL_COMPILERS':'compiler-final'})
run([sys.executable,'-m','unittest','tests.test_scalar_reduce_source','-v'],{'NANOLANG_REDUCE_COMPILERS':str(work/'compiler-final')})
run([sys.executable,'-m','unittest','tests.test_union_aggregate_payloads','tests.test_selfhost_union_storage','tests.test_selfhost_layout_validation','-v'],{'NANO_GENERIC_CSEED_COMPILER':str(work/'compiler-final'),'NANOLANG_SELFHOST_COMPILER':str(work/'compiler-final'),'NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang'})
manifest['inputs_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest[name] for name in ['src/nanoisa/nvm2c.c','Makefile.gnu']) and hashlib.sha256(module.read_bytes()).hexdigest()==manifest['module_sha256']
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
