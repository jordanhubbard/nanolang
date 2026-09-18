import pathlib,subprocess,sys,json,hashlib,shlex,time,os
root=pathlib.Path.cwd();out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files=subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines()
def inventory():return {p:digest(p) for p in files if pathlib.Path(p).is_file()}
(out/'before.json').write_text(json.dumps(inventory(),indent=2)+'\n')
compilers=['/usr/bin/gcc','/usr/bin/clang'] if sys.platform!='darwin' else ['/opt/homebrew/opt/llvm/bin/clang']
tools={p:{'sha256':digest(p),'version':subprocess.check_output([p,'--version'],text=True)} for p in compilers}
(out/'tools.json').write_text(json.dumps(tools,indent=2)+'\n')
results=[]
def run(name,cmd,env=None):
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:
  f.write(shlex.join(cmd)+'\n');f.flush();p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 results.append({'name':name,'status':p.returncode,'seconds':time.monotonic()-start,'command':cmd})
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
try:
 run('target',['make','-j8','CC='+compilers[0],'test-nanocore-export-buffer'])
 plan=subprocess.check_output(['make','-Bn','CC='+compilers[0],'test-nanocore-export-buffer'],text=True)
 line=next(l for l in plan.splitlines() if l.startswith(compilers[0]+' ') and 'tests/test_nanocore_export_buffer.c' in l)
 original=shlex.split(line)
 for index,cc in enumerate(compilers):
  cmd=[cc]+[x for x in original[1:] if not x.startswith('-O')]
  target=out/('buffer-'+str(index));cmd[cmd.index('-o')+1]=str(target)
  cmd[1:1]=['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
  if sys.platform!='darwin' and 'clang' in cc:cmd.insert(1,'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')
  run('compile-'+str(index),cmd)
  run('sanitizer-'+str(index),[str(target)],{**os.environ,'ASAN_OPTIONS':'detect_leaks=1:abort_on_error=1'})
finally:
 (out/'after.json').write_text(json.dumps(inventory(),indent=2)+'\n')
 (out/'tools-after.json').write_text(json.dumps({p:digest(p) for p in compilers},indent=2)+'\n')
 (out/'environment.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'platform':sys.platform,'uname':list(os.uname())},indent=2)+'\n')
