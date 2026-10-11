from pathlib import Path
import hashlib,json,os,subprocess,time,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_one_ir_compiler import OneIrCompiler, HOST_RUNTIME
root=Path.cwd();work=Path('/private/tmp/nl51-native-effects')
inputs=['src/nanoisa/nvm2c_effects.h', 'src/nanoisa/nvm2c_effect_flow.h', 'src/nanoisa/nvm2c_effect_emit.h', 'src/nanoisa/nvm2c_callables.c', 'tests/test_native_effects.py', 'tests/test_selfhost_effects.py', 'src/module_metadata.c', 'src/native_type_key.h', 'src/c_backend.c', 'src/parser.c', 'src/nominal_types.c', 'src_nano/compiler/lexer.nano', 'src_nano/compiler/effect_declarations.nano', 'src_nano/generated/compiler_schema.nano', 'src/nanoisa/assembler.c', 'schema/compiler_schema.json', 'src/generated/compiler_schema.h', 'src_nano/generated/compiler_ast.nano', 'src_nano/parser.nano', 'src_nano/typecheck.nano', 'src_nano/nanoc_v06.nano', 'src_nano/compiler/nanoisa_codegen.nano', 'src_nano/compiler/nominal_bindings.nano', 'src/nanolang.h', 'src/env.c', 'src/typechecker.c', 'src/nanovirt/codegen.c', 'src/nanovm/vm.c', 'src/nanoisa/nvm2c.c']
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
run([root/'bin/nanoc_c',root/'src_nano/nanoc_v06.nano','-o',work/'seed'],{'NANO_SHADOW_TRACE':'1'})
run([work/'seed',root/'src_nano/nanoc_v06.nano','--emit-nvm','-o',work/'compiler.nvm'])
run([root/'bin/nano_vm','--verify-only',work/'compiler.nvm'])
run([root/'bin/nvm2c',work/'compiler.nvm','-o',work/'compiler.c'])
OneIrCompiler().assert_no_vm_wrapper((work/'compiler.c').read_text())
run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',work/'compiler.c','-o',work/'compiler',*HOST_RUNTIME])
run([work/'compiler','--help'])
run([sys.executable,'tests/test_selfhost_effects.py'],{'NANOLANG_EFFECT_COMPILERS':os.pathsep.join([str(root/'bin/nano_virt'),str(work/'seed'),str(work/'compiler')]),'NANOLANG_EFFECT_TARGETS':'vm,native','NANOLANG_EFFECT_CC':'/opt/homebrew/opt/llvm/bin/clang','NANOLANG_EFFECT_CFLAGS':'-fsanitize=address,undefined -fno-sanitize-recover=all -g','ASAN_OPTIONS':'detect_leaks=1'})
run([sys.executable,'tests/test_effect_binding_metadata.py'],{'NANOLANG_EFFECT_NAME_COMPILER':str(work/'compiler'),'NANOLANG_EFFECT_NAME_REQUIRE_DEBUG':'0'})
run([sys.executable,'-m','unittest','tests.test_selfhost_opaque_imports','tests.test_selfhost_union_storage','tests.test_selfhost_layout_validation','-v'],{'NANOLANG_OPAQUE_COMPILER':str(work/'compiler'),'NANOLANG_SELFHOST_COMPILER':str(work/'compiler')})
manifest['artifacts']={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ('seed','compiler.nvm','compiler.c','compiler')}
manifest['inputs_unchanged']=manifest['inputs']==hashes()
(work/'product-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert manifest['inputs_unchanged']
