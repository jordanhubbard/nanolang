from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-generic/native')
inputs=['src/typechecker.c','src/nanoisa/nvm_format.c','src/nanoisa/nvm_format.h','src/nanoisa/assembler.c','src/nanovirt/codegen.c','src/nanovirt/generic_codegen.inc','src_nano/compiler/module_bindings.nano','src_nano/compiler/nanoisa_codegen.nano']
def hashes():return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs}
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':hashes(),'steps':[],'seed_build_log':'../seed-native-build.log'}
def run(args,env=None,timeout=900):
 index=len(manifest['steps']);start=time.monotonic()
 with (work/f'product-{index:02d}.log').open('w') as log:
  result=subprocess.run(list(map(str,args)),cwd=root,env={**os.environ,'NANOLANG_SHADOW_TIMEOUT_SECONDS':'60',**(env or {})},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 manifest['steps'].append({'argv':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-start,'log':f'product-{index:02d}.log'})
 (work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(index,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run([sys.executable,'-m','unittest','tests.test_generic_binding_metadata','-v'],{'NANOLANG_GENERIC_METADATA_COMPILERS':os.pathsep.join([str(root/'bin/nano_virt'),str(work.parent/'seed-native')]),'CC':'/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1'})
run([work.parent/'seed-native',root/'src_nano/nanoc_v06.nano','--emit-nvm','-o',work/'compiler.nvm'])
run([root/'bin/nano_vm','--verify-only',work/'compiler.nvm'])
run([root/'bin/nvm2c',work/'compiler.nvm','-o',work/'compiler.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'compiler.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'compiler.c','-o',work/'compiler',*HOST_RUNTIME])
run([work/'compiler','--help'])
run([sys.executable,'-m','unittest','tests.test_generic_binding_metadata','-v'],{'NANOLANG_GENERIC_METADATA_COMPILERS':str(work/'compiler'),'CC':'/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1'})
run([sys.executable,'-m','unittest','tests.test_function_purity_metadata','tests.test_generic_union_parameters','tests.test_contextual_generic_functions','-v'],{'NANOLANG_PURITY_COMPILERS':str(work/'compiler'),'NANO_GENERIC_CSEED_COMPILER':str(work/'compiler'),'NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','ASAN_OPTIONS':'detect_leaks=1'})
manifest['inputs_unchanged']=manifest['inputs']==hashes()
manifest['artifacts']={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ['compiler.nvm','compiler.c','compiler']}
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
