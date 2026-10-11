from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-upvalue-names')
inputs=['src/nanoisa/local_bindings.c','src/nanoisa/local_bindings.h','src/nanoisa/assembler.c','src/nanovirt/codegen.c','src_nano/compiler/nanoisa_codegen.nano']
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
run([sys.executable,'-m','unittest','tests.test_local_binding_metadata','-v'],{'NANOLANG_LOCAL_NAME_COMPILERS':str(work/'compiler'),'CC':'/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1'})
run([sys.executable,'-m','unittest','tests.test_native_nominal_order','tests.test_selfhost_union_storage','tests.test_selfhost_layout_validation','-v'],{'NANOLANG_NOMINAL_COMPILER_ROOT':str(work),'NANOLANG_NOMINAL_COMPILERS':'compiler','NANOLANG_SELFHOST_COMPILER':str(work/'compiler')})
manifest['inputs_unchanged']=manifest['inputs']==hashes()
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
