import os,pathlib,shlex,subprocess
root=pathlib.Path('/Users/jordanh/Src/nanolang')
lines=pathlib.Path('/private/tmp/nl51-websocket-hosted-clang.log').read_text().splitlines()
commands=[shlex.split(line) for line in lines if line.startswith('/opt/') and 'tests/nanoisa/test_websocket_hosted.c' in line]
assert len(commands)==2
for label,cc,sanitize in [('gcc','/opt/homebrew/bin/gcc-16',False),('sanitized','/opt/homebrew/opt/llvm/bin/clang',True)]:
 out=pathlib.Path('/private/tmp/nl51-websocket-hosted-'+label);out.mkdir(exist_ok=True)
 flags=['-std=c11','-D_GNU_SOURCE','-Wall','-Wextra','-Werror','-g','-O1']
 if sanitize:flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
 env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
 with (out/'result.log').open('w') as log:
  def run(cmd):
   log.write(shlex.join(cmd)+'\n');log.flush()
   r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
   log.write('exit='+str(r.returncode)+'\n');log.flush()
   if r.returncode:raise SystemExit(r.returncode)
  replacements={}
  for source in ['nvm_v2_convert','nvm_v2_module','service_bindings_module']:
   obj=str(out/(source+'.o'));run([cc,*flags,'-c','src/nanoisa/'+source+'.c','-o',obj]);replacements['obj/nanoisa/'+source+'.o']=obj
  for n,original in enumerate(commands):
   cmd=[cc,*original[1:]]
   cmd=[replacements.get(arg,arg) for arg in cmd]
   if sanitize:cmd[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
   cmd[cmd.index('-o')+1]=str(out/('probe'+str(n)))
   run(cmd);run([cmd[cmd.index('-o')+1]])
 print(label+' passed',flush=True)
