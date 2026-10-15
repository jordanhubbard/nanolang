from pathlib import Path
import json,os,subprocess,time
root=Path.cwd();work=Path('/private/tmp/nl51-product-current-h8l42nxe/selfhost-followthrough');work.mkdir(exist_ok=True)
module=work.parent/'nano-one-ir-compiler-d3fe66ju/rebuilt-compiler.nvm'
source,compiler=work/'compiler.c',work/'compiler'
hello_module,hello_source,hello_native=work/'hello.nvm',work/'hello.c',work/'hello'
commands=[([root/'bin/nvm2c',module,'-o',source],240),(['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',source,'-o',compiler,root/'bin/nano_aot_runtime.o','-lm'],240),([compiler,'--help'],10),(['python3','-m','unittest','tests.test_selfhost_captures','tests.test_native_mutable_record_arrays','tests.test_genenv_scope'],180),([compiler,root/'examples/language/nl_hello.nano','--emit-nvm','-o',hello_module],180),([root/'bin/nano_vm',hello_module],10),([root/'bin/nvm2c',hello_module,'-o',hello_source],180),(['cc','-std=c11','-Wall','-Wextra','-Werror',hello_source,'-o',hello_native],180),([hello_native],10)]
manifest={'module':str(module),'steps':[]}
for index,(args,timeout) in enumerate(commands):
 command=list(map(str,args));print('I run',index,command,flush=True);start=time.monotonic()
 env={**os.environ,'NANOLANG_SELFHOST_COMPILER':str(compiler),'NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_MODULE_PATH':str(root/'modules')}
 with (work/f'{index:02d}.stdout').open('wb') as out,(work/f'{index:02d}.stderr').open('wb') as err:
  result=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=err,timeout=timeout)
 manifest['steps'].append({'argv':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start})
 (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
 if index in (5,8):assert (work/f'{index:02d}.stdout').read_bytes()==b'Hello from NanoLang!\n'
print('I passed the fresh self-hosted NanoISA compiler product and its acceptance corpus.',flush=True)
