from pathlib import Path
import os,subprocess,tempfile
root=Path.cwd()
compiler=['bin/nano_vm','/private/tmp/nanolang-import-callable-corrected-compiler.nvm','--']
cc='/opt/homebrew/opt/llvm/bin/clang'
with tempfile.TemporaryDirectory(prefix='nano-product-link-flags-') as tmp:
 work=Path(tmp);(work/'bin').mkdir()
 runtime=work/'runtime.c';runtime.write_text('int covered_answer(void) { return 42; }\n')
 subprocess.run([cc,'--coverage','-c',runtime,'-o',work/'bin/nano_aot_runtime.o'],check=True)
 source=work/'hello.nano';source.write_text('fn main()->int{return 0}\nshadow main {assert (== (main) 0)}\n')
 for name,flags in [('fallback',{'LDFLAGS':'--coverage'}),('override',{'LDFLAGS':'-lmissing_outer_link_flag','NANO_LDFLAGS':'--coverage'}),('missing',{})]:
  env=dict(os.environ)
  for key in ('NANO_LDFLAGS','LDFLAGS','NANO_CFLAGS'):env.pop(key,None)
  env.update(flags);env.update(NANOLANG_ROOT=str(work),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_VM=str(root/'bin/nano_vm'),CC=cc,NANO_CC=cc)
  output=work/name;output.write_bytes(b'prior-output')
  result=subprocess.run(compiler+[str(source),'-o',str(output)],cwd=root,env=env,capture_output=True,text=True,timeout=120)
  print(name,'compile exit',result.returncode,'preserved',output.read_bytes()==b'prior-output',flush=True)
  print(result.stdout+result.stderr,flush=True)
  if result.returncode==0:print(name,'execution exit',subprocess.run([output],cwd=work,env=env).returncode,flush=True)
