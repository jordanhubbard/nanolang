from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-context-storage')
inputs=['src/typechecker.c','src_nano/typecheck.nano','src_nano/compiler/nanoisa_codegen.nano','src/nanoisa/nvm2c.c']
def hashes():return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs}
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':hashes(),'steps':[]}
def run(args,env=None,timeout=600):
 index=len(manifest['steps']);start=time.monotonic()
 with (work/f'product-{index:02d}.log').open('w') as log:
  result=subprocess.run(list(map(str,args)),cwd=root,env={**os.environ,**(env or {})},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 manifest['steps'].append({'argv':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-start,'log':f'product-{index:02d}.log'})
 (work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(index,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run([work/'seed',root/'src_nano/nanoc_v06.nano','--emit-nvm','-o',work/'compiler.nvm'])
run([root/'bin/nano_vm','--verify-only',work/'compiler.nvm'])
run([root/'bin/nvm2c',work/'compiler.nvm','-o',work/'compiler.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'compiler.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'compiler.c','-o',work/'compiler',*HOST_RUNTIME])
run([work/'compiler','--help'])
run([sys.executable,'-m','unittest','tests.test_conditional_generic_callbacks','tests.test_contextual_generic_functions','tests.test_union_aggregate_payloads','tests.test_native_union_tuples','-v'],{'NANO_GENERIC_CSEED_COMPILER':str(work/'compiler'),'NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang'})
run([sys.executable,'-m','unittest','tests.test_native_nominal_order','tests.test_selfhost_union_storage','tests.test_selfhost_layout_validation','-v'],{'NANOLANG_NOMINAL_COMPILER_ROOT':str(work),'NANOLANG_NOMINAL_COMPILERS':'compiler','NANOLANG_SELFHOST_COMPILER':str(work/'compiler')})
manifest['inputs_unchanged']=manifest['inputs']==hashes()
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
