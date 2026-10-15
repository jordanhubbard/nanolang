from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-effect-conflicts/product-audit')
inputs=['src/nanolang.h','src/typechecker.c','src_nano/typecheck.nano','src/nanovirt/codegen.c','src/nanovirt/generic_codegen.inc','src_nano/compiler/nanoisa_codegen.nano','tests/test_effect_signature_identity.py','src/nanoisa/nvm2c.c','tests/test_union_arrays.py','tests/test_imported_effect_identity.py','src/env.c','src/module.c','src/parser.c','src/transpiler.c','src/transpiler_iterative_v3_twopass.c','src_nano/nanoc_v06.nano','src_nano/parser.nano','src_nano/compiler/effect_declarations.nano','src_nano/compiler/effect_bindings.nano']
def hashes():return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in inputs}
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':hashes(),'steps':[],'seed_provenance':'Previously qualified native compiler from imported-effects-20261011/product-audit; source checkpoint 60e6a0ae1'}
def run(args,env=None,timeout=900):
 index=len(manifest['steps']);start=time.monotonic()
 with (work/f'product-{index:02d}.log').open('w') as log:
  result=subprocess.run(list(map(str,args)),cwd=root,env={**os.environ,'NANO_SHADOW_TIMEOUT_SECONDS':'60',**(env or {})},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 manifest['steps'].append({'argv':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-start,'log':f'product-{index:02d}.log'})
 (work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(index,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
run([sys.executable,'-m','unittest','tests.test_selfhost_effects','tests.test_effect_signature_identity','tests.test_union_arrays','tests.test_imported_effect_identity','-v'],{'NANOLANG_EFFECT_COMPILERS':str(root/'bin/nano_virt'),'NANOLANG_EFFECT_SIGNATURE_COMPILERS':str(root/'bin/nano_virt'),'NANOLANG_IMPORTED_EFFECT_COMPILERS':str(root/'bin/nano_virt'),'NANOLANG_UNION_ARRAY_COMPILERS':str(root/'bin/nano_virt'),'NANOLANG_EFFECT_TARGETS':'vm,native','NANOLANG_EFFECT_CC':'/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1'})
run([work/'seed',root/'src_nano/nanoc_v06.nano','--emit-nvm','-o',work/'compiler.nvm'])
run([root/'bin/nano_vm','--verify-only',work/'compiler.nvm'])
run([root/'bin/nvm2c',work/'compiler.nvm','-o',work/'compiler.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'compiler.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'compiler.c','-o',work/'compiler',*HOST_RUNTIME])
run([work/'compiler','--help'])
run([sys.executable,'-m','unittest','tests.test_selfhost_effects','tests.test_effect_signature_identity','tests.test_union_arrays','tests.test_imported_effect_identity','-v'],{'NANOLANG_EFFECT_COMPILERS':str(work/'compiler'),'NANOLANG_EFFECT_SIGNATURE_COMPILERS':str(work/'compiler'),'NANOLANG_IMPORTED_EFFECT_COMPILERS':str(work/'compiler'),'NANOLANG_UNION_ARRAY_COMPILERS':str(work/'compiler'),'NANOLANG_EFFECT_TARGETS':'vm,native','NANOLANG_EFFECT_CC':'/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1'})
run([sys.executable,'-m','unittest','tests.test_generic_binding_metadata','tests.test_function_purity_metadata','tests.test_generic_union_parameters','tests.test_contextual_generic_functions','-v'],{'NANOLANG_GENERIC_METADATA_COMPILERS':str(work/'compiler'),'NANOLANG_PURITY_COMPILERS':str(work/'compiler'),'NANO_GENERIC_CSEED_COMPILER':str(work/'compiler'),'NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','ASAN_OPTIONS':'detect_leaks=1'})
manifest['inputs_unchanged']=manifest['inputs']==hashes()
manifest['artifacts']={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ['seed','compiler.nvm','compiler.c','compiler']}
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
